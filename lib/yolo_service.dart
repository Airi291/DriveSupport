import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class YoloService {
  Interpreter? _interpreter;
  final int inputSize;
  final double scoreThreshold;

  YoloService({this.inputSize = 640, this.scoreThreshold = 0.5});

  Future<void> loadModel() async {
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/models/best_float32.tflite',
      options: options,
    );
  }

  bool get isReady => _interpreter != null;

  int runOnBgraFrame(Uint8List bgraBytes, int width, int height) {
    final itp = _interpreter!;
    final src = _bgraToImage(bgraBytes, width, height);
    final letterboxed = _letterbox(src, inputSize, inputSize);

    final input = List.generate(
      1,
      (_) => List.generate(inputSize, (_) => List.filled(inputSize * 3, 0.0)),
    );

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final p = letterboxed.getPixel(x, y);
        input[0][y][x * 3 + 0] = p.r / 255.0;
        input[0][y][x * 3 + 1] = p.g / 255.0;
        input[0][y][x * 3 + 2] = p.b / 255.0;
      }
    }

    final output = <int, Object>{};
    final outputTensors = itp.getOutputTensors();
    for (var i = 0; i < outputTensors.length; i++) {
      final shape = outputTensors[i].shape;
      final flat = List.filled(shape.reduce((a, b) => a * b), 0.0);
      output[i] = flat.toShape(shape);
    }

    itp.runForMultipleInputs([input], output);

    int count = 0;
    for (final obj in output.values) {
      count += _countDetections(obj, scoreThreshold);
    }
    return count;
  }

  img.Image _bgraToImage(Uint8List bgra, int w, int h) {
    final out = img.Image(width: w, height: h);
    int j = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final b = bgra[j++];
        final g = bgra[j++];
        final r = bgra[j++];
        j++;
        out.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return out;
  }

  img.Image _letterbox(img.Image src, int tw, int th) {
    final srcW = src.width, srcH = src.height;
    final scale = (srcW / srcH > tw / th) ? tw / srcW : th / srcH;
    final nw = (srcW * scale).round();
    final nh = (srcH * scale).round();

    final resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear,
    );
    final canvas = img.Image(width: tw, height: th);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    final ox = ((tw - nw) / 2).round();
    final oy = ((th - nh) / 2).round();
    img.compositeImage(canvas, resized, dstX: ox, dstY: oy);
    return canvas;
  }

  int _countDetections(Object tensor, double thr) {
    int cnt = 0;
    void visit(dynamic t) {
      if (t is List) {
        if (t.isNotEmpty && t.first is num) {
          final row = t.cast<num>();
          final score = row.reduce((a, b) => a > b ? a : b).toDouble();
          if (score >= thr) cnt++;
        } else {
          for (final e in t) visit(e);
        }
      }
    }

    visit(tensor);
    return cnt;
  }
}

extension ReshapeList on List {
  dynamic toShape(List<int> shape) {
    int total = shape.reduce((a, b) => a * b);
    if (length != total) return this;
    dynamic build(List<int> dims, int offset) {
      if (dims.isEmpty) return this[offset];
      final size = dims.first;
      final step = dims.skip(1).fold(1, (a, b) => a * b);
      return List.generate(
        size,
        (i) => build(dims.skip(1).toList(), offset + i * step),
      );
    }

    return build(shape, 0);
  }
}
