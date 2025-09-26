import 'package:flutter/material.dart';
import 'package:camera_macos/camera_macos_view.dart';
import 'package:camera_macos/camera_macos_arguments.dart';
import 'package:camera_macos/camera_macos_controller.dart';

void main() {
  runApp(const MyApp());
}

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
  String? _status = "カメラを初期化中...";
  CameraMacOSController? _controller;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: CameraMacOSView(
          cameraMode: CameraMacOSMode.photo,
          onCameraInizialized: (controller) {
            debugPrint("✅ Camera initialized: $controller");
            setState(() {
              _controller = controller;
              _status = "カメラ準備OK ✅";
            });
          },
        ),
      ),
      bottomNavigationBar: _status != null
          ? Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(_status!, textAlign: TextAlign.center),
            )
          : null,
    );
  }
}
