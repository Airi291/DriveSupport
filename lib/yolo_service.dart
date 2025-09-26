// lib/yolo_service.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Detection {
  final int classId;
  final double score;
  final RectF box;
  Detection({required this.classId, required this.score, required this.box});
}

class RectF {
  final double left, top, right, bottom;
  const RectF(this.left, this.top, this.right, this.bottom);

  double get width => right - left;
  double get height => bottom - top;

  RectF intersect(RectF other) {
    final l = math.max(left, other.left);
    final t = math.max(top, other.top);
    final r = math.min(right, other.right);
    final b = math.min(bottom, other.bottom);
    if (r <= l || b <= t) return const RectF(0, 0, 0, 0);
    return RectF(l, t, r, b);
  }

  double area() => math.max(0, width) * math.max(0, height);

  double iou(RectF other) {
    final inter = intersect(other).area();
    final uni = area() + other.area() - inter;
    return uni <= 0 ? 0 : inter / uni;
  }
}

class LetterboxResult {
  final img.Image image;
  final double scale;
  final int padX;
  final int padY;
  const LetterboxResult(this.image, this.scale, this.padX, this.padY);
}

class YoloService {
  Interpreter? _itp;

  /// ★ まずは 320 で大幅に軽くなります（必要なら 416/512 に上げる）
  final int inputSize;
  final double scoreThreshold;
  final double iouThreshold;

  // --- 再利用するバッファ ---
  Float32List? _inF32; // [1, H, W, 3]
  Float32List? _outF32; // モデル出力の生バッファ
  List<int>? _outShape; // 出力テンソル形状キャッシュ

  YoloService({
    this.inputSize = 640,
    this.scoreThreshold = 0.40,
    this.iouThreshold = 0.45,
  });

  Future<void> loadModel({
    String modelAsset = 'assets/models/best_float32.tflite',
    String? labelsAsset,
  }) async {
    // CPU最適化：XNNPACK + マルチスレッド（1コア余らせ）
    final cpuThreads = math.max(1, Platform.numberOfProcessors - 1);
    final options = InterpreterOptions()
      ..threads = cpuThreads
      ..addDelegate(XNNPackDelegate());

    _itp = await Interpreter.fromAsset(modelAsset, options: options);
    _allocBuffers();
    _printModelInfo();
  }

  void _allocBuffers() {
    final itp = _itp!;
    // 入力は [1, inputSize, inputSize, 3] の float32 を想定
    _inF32 = Float32List(inputSize * inputSize * 3);

    // 出力テンソル0の形状から必要サイズを確保
    final outT = itp.getOutputTensors().first;
    _outShape = List<int>.from(outT.shape);
    final outCount = _outShape!.fold<int>(1, (a, b) => a * b);
    _outF32 = Float32List(outCount);
  }

  bool get isReady => _itp != null;

  void _printModelInfo() {
    final itp = _itp!;
    for (final t in itp.getOutputTensors()) {
      print(
        'YOLO output tensor: name=${t.name} shape=${t.shape} type=${t.type}',
      );
    }
  }

  /// BGRA(32bpp)フレーム → 検出結果（全処理をバッファ直叩きで高速化）
  List<Detection> runOnBgraFrame(Uint8List bgra, int width, int height) {
    final itp = _itp!;
    final inF32 = _inF32!;
    final outF32 = _outF32!;
    final outShape = _outShape!;

    // 1) BGRA → Image
    final src = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: bgra.buffer,
      numChannels: 4,
      order: img.ChannelOrder.bgra,
      rowStride: width * 4,
    );

    // 2) letterbox（inputSize四方）
    final prep = _letterboxToSquare(src, inputSize);

    // 3) RGB bytes を一括取得 → Float32に正規化して詰める
    final rgb = prep.image.getBytes(order: img.ChannelOrder.rgb);
    final len = inputSize * inputSize;
    // NHWC: ピクセル順で R,G,B を詰める
    // idx: [pixelIndex*3 + channel]
    int di = 0;
    for (int i = 0; i < len; i++) {
      // rgb は R,G,B の順
      final r = rgb[3 * i + 0] / 255.0;
      final g = rgb[3 * i + 1] / 255.0;
      final b = rgb[3 * i + 2] / 255.0;
      inF32[di++] = r;
      inF32[di++] = g;
      inF32[di++] = b;
    }

    // 4) 推論（★ 形を合わせた多次元 List を渡す）

    // 入力 [1, H, W, 3] を1回だけ確保して再利用（クラスのフィールドにしてもOK）
    final input4d = List.generate(
      1,
      (_) => List.generate(
        inputSize,
        (_) => List.generate(
          inputSize,
          (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );

    // inF32 から input4d へ詰める（1回の for でOK）
    {
      int i = 0;
      for (int y = 0; y < inputSize; y++) {
        final row = input4d[0][y];
        for (int x = 0; x < inputSize; x++) {
          final px = row[x];
          px[0] = inF32[i++]; // R
          px[1] = inF32[i++]; // G
          px[2] = inF32[i++]; // B
        }
      }
    }

    // 出力 [1, 25, 8400] を用意（C,N は getOutputTensors()[0].shape から）
    final c = _outShape![1]; // 25
    final n = _outShape![2]; // 8400
    final output3d = List.generate(
      1,
      (_) => List.generate(
        c,
        (_) => List<double>.filled(n, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    // 実行
    itp.runForMultipleInputs([input4d], {0: output3d});

    // 5) 出力パース（[1, C, N]）を直接読む
    final preds = <_Pred>[];
    final out = output3d[0]; // [C][N]
    final numClasses = c - 5;
    for (int idx = 0; idx < n; idx++) {
      final cx = out[0][idx].toDouble();
      final cy = out[1][idx].toDouble();
      final w = out[2][idx].toDouble();
      final h = out[3][idx].toDouble();
      final obj = out[4][idx].toDouble();

      int bestCls = 0;
      double bestScore = 0.0;
      for (int cls = 0; cls < numClasses; cls++) {
        final s = out[5 + cls][idx].toDouble();
        if (s > bestScore) {
          bestScore = s;
          bestCls = cls;
        }
      }
      final score = obj * bestScore;
      if (score < scoreThreshold) continue;

      final left = cx - w / 2,
          top = cy - h / 2,
          right = cx + w / 2,
          bottom = cy + h / 2;
      final mapped = _mapBoxBackToSrc(
        RectF(left, top, right, bottom),
        prep.scale,
        prep.padX,
        prep.padY,
        src.width,
        src.height,
      );
      preds.add(_Pred(classId: bestCls, score: score, rect: mapped));
    }

    // 6) NMS→Detection
    final kept = _nms(preds, iouThreshold);
    return kept
        .map((p) => Detection(classId: p.classId, score: p.score, box: p.rect))
        .toList();
  }

  // ------ 出力パース（[N, C] か [1, C, N] の両方に対応） ------
  List<_Pred> _parseDetections(
    Float32List out,
    List<int> shape,
    LetterboxResult prep,
    int srcW,
    int srcH,
  ) {
    final preds = <_Pred>[];

    if (shape.length == 2) {
      // 例: [8400, 25] = [N, C]
      final N = shape[0];
      final C = shape[1];
      final numClasses = C - 5;
      for (int n = 0; n < N; n++) {
        final base = n * C;
        final cx = out[base + 0].toDouble();
        final cy = out[base + 1].toDouble();
        final w = out[base + 2].toDouble();
        final h = out[base + 3].toDouble();
        final obj = out[base + 4].toDouble();

        int bestCls = 0;
        double bestScore = 0.0;
        for (int c = 0; c < numClasses; c++) {
          final s = out[base + 5 + c].toDouble();
          if (s > bestScore) {
            bestScore = s;
            bestCls = c;
          }
        }
        final score = obj * bestScore;
        if (score < scoreThreshold) continue;

        final left = cx - w / 2,
            top = cy - h / 2,
            right = cx + w / 2,
            bottom = cy + h / 2;
        final mapped = _mapBoxBackToSrc(
          RectF(left, top, right, bottom),
          prep.scale,
          prep.padX,
          prep.padY,
          srcW,
          srcH,
        );
        preds.add(_Pred(classId: bestCls, score: score, rect: mapped));
      }
    } else if (shape.length == 3 && shape[0] == 1) {
      // 例: [1, 25, 8400] = [1, C, N]（チャネル優先）
      final C = shape[1];
      final N = shape[2];
      final numClasses = C - 5;
      // out のインデックス: out[c*N + n]
      for (int n = 0; n < N; n++) {
        final cx = out[0 * N + n].toDouble();
        final cy = out[1 * N + n].toDouble();
        final w = out[2 * N + n].toDouble();
        final h = out[3 * N + n].toDouble();
        final obj = out[4 * N + n].toDouble();

        int bestCls = 0;
        double bestScore = 0.0;
        for (int c = 0; c < numClasses; c++) {
          final s = out[(5 + c) * N + n].toDouble();
          if (s > bestScore) {
            bestScore = s;
            bestCls = c;
          }
        }
        final score = obj * bestScore;
        if (score < scoreThreshold) continue;

        final left = cx - w / 2,
            top = cy - h / 2,
            right = cx + w / 2,
            bottom = cy + h / 2;
        final mapped = _mapBoxBackToSrc(
          RectF(left, top, right, bottom),
          prep.scale,
          prep.padX,
          prep.padY,
          srcW,
          srcH,
        );
        preds.add(_Pred(classId: bestCls, score: score, rect: mapped));
      }
    }
    return preds;
  }

  // ------ ユーティリティ ------

  LetterboxResult _letterboxToSquare(img.Image src, int size) {
    final sw = src.width.toDouble();
    final sh = src.height.toDouble();
    final scale = math.min(size / sw, size / sh);
    final nw = (sw * scale).round();
    final nh = (sh * scale).round();

    final resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear,
    );
    final canvas = img.Image(width: size, height: size);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    final padX = ((size - nw) / 2).round();
    final padY = ((size - nh) / 2).round();
    img.compositeImage(canvas, resized, dstX: padX, dstY: padY);
    return LetterboxResult(canvas, scale, padX, padY);
  }

  RectF _mapBoxBackToSrc(
    RectF box640,
    double scale,
    int padX,
    int padY,
    int srcW,
    int srcH,
  ) {
    final left = (box640.left - padX) / scale;
    final top = (box640.top - padY) / scale;
    final right = (box640.right - padX) / scale;
    final bottom = (box640.bottom - padY) / scale;

    final double l = left.clamp(0.0, srcW.toDouble()).toDouble();
    final double t = top.clamp(0.0, srcH.toDouble()).toDouble();
    final double r = right.clamp(0.0, srcW.toDouble()).toDouble();
    final double b = bottom.clamp(0.0, srcH.toDouble()).toDouble();
    return RectF(l, t, r, b);
  }

  List<_Pred> _nms(List<_Pred> boxes, double iouThr) {
    boxes.sort((a, b) => b.score.compareTo(a.score));
    final kept = <_Pred>[];
    for (final b in boxes) {
      bool keep = true;
      for (final k in kept) {
        if (b.classId == k.classId && b.rect.iou(k.rect) > iouThr) {
          keep = false;
          break;
        }
      }
      if (keep) kept.add(b);
    }
    return kept;
  }
}

class _Pred {
  final int classId;
  final double score;
  final RectF rect;
  _Pred({required this.classId, required this.score, required this.rect});
}
