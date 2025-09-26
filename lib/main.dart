import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
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

  final _yolo = YoloService(inputSize: 640, scoreThreshold: 0.50);
  bool _modelReady = false;

  Timer? _timer;
  bool _busy = false;
  int _lastCount = 0;

  @override
  void initState() {
    super.initState();
    _yolo.loadModel().then((_) {
      setState(() {
        _modelReady = true;
      });
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.destroy();
    super.dispose();
  }

  void _startLoop() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 300), (_) async {
      if (!_modelReady || _controller == null || _busy) return;
      _busy = true;
      try {
        final res = await _controller!.takePicture();
        if (res == null || res.url == null) {
          debugPrint("⚠️ takePicture() returned null or url is null");
          return;
        }

        final file = File(res.url!);
        if (!await file.exists()) {
          debugPrint("⚠️ file not found: ${res.url}");
          return;
        }

        final bytes = await file.readAsBytes();
        final decoded = img.decodeTiff(bytes) ?? img.decodeImage(bytes);
        if (decoded == null) {
          debugPrint("⚠️ decode failed");
          return;
        }

        final rgba = decoded.convert(numChannels: 4);

        // RGBに変換
        final rgbBytes = Uint8List(rgba.width * rgba.height * 3);
        int j = 0;
        for (int y = 0; y < rgba.height; y++) {
          for (int x = 0; x < rgba.width; x++) {
            final p = rgba.getPixel(x, y);
            rgbBytes[j++] = p.r.toInt();
            rgbBytes[j++] = p.g.toInt();
            rgbBytes[j++] = p.b.toInt();
          }
        }

        // BGRAに変換（YOLO入力用）
        final bgra = Uint8List(rgba.width * rgba.height * 4);
        int k = 0, m = 0;
        for (; k < rgbBytes.length;) {
          final r = rgbBytes[k++], g = rgbBytes[k++], b = rgbBytes[k++];
          bgra[m++] = b;
          bgra[m++] = g;
          bgra[m++] = r;
          bgra[m++] = 255;
        }

        final count = _yolo.runOnBgraFrame(bgra, rgba.width, rgba.height);
        if (mounted) {
          setState(() => _lastCount = count);
        }
      } catch (e) {
        debugPrint('inference error: $e');
      } finally {
        _busy = false;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
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
                _modelReady ? 'Detections: $_lastCount' : 'モデル読み込み中…',
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
