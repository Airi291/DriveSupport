// lib/main.dart
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

// ↓ camera_macos がプロジェクト内にある前提
import 'package:camera_macos/camera_macos_view.dart';
import 'package:camera_macos/camera_macos_arguments.dart';
import 'package:camera_macos/camera_macos_controller.dart';

import 'package:image/image.dart' as img;
import 'yolo_service.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraPage(),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});
  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  String _status = "カメラを初期化中...";
  CameraMacOSController? _controller;

  final _yolo = YoloService(inputSize: 640, scoreThreshold: 0.40);
  bool _modelReady = false;

  Timer? _timer;
  bool _busy = false;

  List<Detection> _detections = [];
  int _lastW = 0, _lastH = 0; // 元画像サイズ（枠描画のため）

  @override
  void initState() {
    super.initState();
    _initYolo();
  }

  Future<void> _initYolo() async {
    await _yolo.loadModel(
      modelAsset: 'assets/models/best_float32.tflite',
      // labelsAsset: 'assets/labels_mtsd21.txt', // 必要に応じて使う
    );
    setState(() => _modelReady = true);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.destroy();
    super.dispose();
  }

  void _startLoop() {
    _timer?.cancel(); // 以降は自前ループに切替

    // 1秒だけ待ってから開始（初回の取りこぼし回避）
    Future<void>(() async {
      await Future.delayed(const Duration(seconds: 1));

      while (mounted) {
        if (_modelReady && _controller != null && !_busy) {
          _busy = true;
          try {
            final res = await _controller!.takePicture();
            if (res == null) {
              debugPrint("⚠️ takePicture() returned null");
              _busy = false;
              await Future.delayed(const Duration(milliseconds: 50));
              continue;
            }

            // まず bytes を優先
            Uint8List? bytes;
            try {
              final b = (res as dynamic).bytes as Uint8List?;
              if (b != null && b.isNotEmpty) bytes = b;
            } catch (_) {}

            // bytes が無いなら url/path から読む
            if (bytes == null) {
              String? path;
              try {
                final url = (res as dynamic).url as String?;
                if (url != null && url.isNotEmpty) {
                  final uri = Uri.parse(url);
                  path = uri.isScheme("file") ? uri.toFilePath() : url;
                }
              } catch (_) {}
              if (path == null || path.isEmpty) {
                try {
                  final p = (res as dynamic).path as String?;
                  if (p != null && p.isNotEmpty) path = p;
                } catch (_) {}
              }
              if (path == null || path.isEmpty) {
                debugPrint("⚠️ neither bytes nor url/path is available");
                _busy = false;
                await Future.delayed(const Duration(milliseconds: 100));
                continue;
              }
              final f = File(path);
              if (!await f.exists()) {
                debugPrint("⚠️ file not found: $path");
                _busy = false;
                await Future.delayed(const Duration(milliseconds: 100));
                continue;
              }
              bytes = await f.readAsBytes();
            }

            // デコード（TIFF優先）
            img.Image? decoded =
                img.decodeTiff(bytes) ?? img.decodeImage(bytes);
            if (decoded == null) {
              debugPrint("⚠️ decode failed");
              _busy = false;
              await Future.delayed(const Duration(milliseconds: 100));
              continue;
            }

            // 先に縮小（計算量とGC削減）——長辺960pxにキャップ
            const maxSide = 960;
            if (decoded.width > maxSide || decoded.height > maxSide) {
              decoded = img.copyResize(
                decoded,
                width: decoded.width >= decoded.height ? maxSide : null,
                height: decoded.height > decoded.width ? maxSide : null,
                interpolation: img.Interpolation.linear,
              );
            }

            _lastW = decoded.width;
            _lastH = decoded.height;

            // RGBA -> BGRA（高速版：getBytesを使用）
            final rgba = decoded.convert(numChannels: 4);
            final bgra = Uint8List.fromList(
              rgba.getBytes(order: img.ChannelOrder.bgra),
            );

            // 推論
            final dets = _yolo.runOnBgraFrame(bgra, rgba.width, rgba.height);
            if (mounted) {
              // 変更があったときだけ setState して再描画負荷を減らす
              if (dets.length != _detections.length) {
                setState(() => _detections = dets);
              } else {
                _detections = dets;
              }
            }
          } catch (e) {
            debugPrint('inference error: $e');
          } finally {
            _busy = false;
          }
        }

        // ここで必ずUIに制御を返す（プレビュー固まり防止）
        await Future.delayed(const Duration(milliseconds: 150));
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final overlay = _detections;
    return Scaffold(
      body: Stack(
        children: [
          Center(
            child: CameraMacOSView(
              cameraMode: CameraMacOSMode.photo,
              onCameraInizialized: (controller) async {
                _controller = controller;
                setState(() => _status = "カメラ準備OK ✅");
                _startLoop();
              },
            ),
          ),
          // 検出枠オーバーレイ
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(painter: _DetPainter(overlay, _lastW, _lastH)),
            ),
          ),
          Positioned(
            top: 12,
            right: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _modelReady ? 'Detections: ${overlay.length}' : 'モデル読み込み中…',
                style: const TextStyle(color: Colors.white, fontSize: 14),
              ),
            ),
          ),
        ],
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Text(_status, textAlign: TextAlign.center),
      ),
    );
  }
}

class _DetPainter extends CustomPainter {
  final List<Detection> dets;
  final int srcW, srcH;
  _DetPainter(this.dets, this.srcW, this.srcH);

  @override
  void paint(Canvas canvas, Size size) {
    if (srcW <= 0 || srcH <= 0) return;
    // カメラのプレビューと元画像のアスペクトが一致しない可能性があるので、
    // とりあえず等倍で描く（黒余白想定）。必要ならここでスケーリング合わせる。
    final scaleX = size.width / srcW;
    final scaleY = size.height / srcH;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = const Color(0xFFFFEB3B); // 黄色（好みで）

    final fill = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0x55FFEB3B);

    final textPainter = (String text) {
      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(
            color: Colors.black,
            backgroundColor: Color(0xAAFFEB3B),
            fontSize: 12,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      return tp;
    };

    for (final d in dets) {
      final l = d.box.left * scaleX;
      final t = d.box.top * scaleY;
      final r = d.box.right * scaleX;
      final b = d.box.bottom * scaleY;
      final rect = Rect.fromLTRB(l, t, r, b);
      canvas.drawRect(rect, paint);
      canvas.drawRect(rect, fill);

      final label = 'id:${d.classId} ${(d.score * 100).toStringAsFixed(1)}%';
      final tp = textPainter(label);
      canvas.drawRect(
        Rect.fromLTWH(l, t - tp.height - 2, tp.width + 6, tp.height + 4),
        Paint()..color = const Color(0xAAFFEB3B),
      );
      tp.paint(canvas, Offset(l + 3, t - tp.height - 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetPainter oldDelegate) {
    return oldDelegate.dets != dets ||
        oldDelegate.srcW != srcW ||
        oldDelegate.srcH != srcH;
  }
}
