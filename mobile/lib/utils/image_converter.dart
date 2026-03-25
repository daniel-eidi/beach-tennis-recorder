import 'dart:typed_data';
import 'dart:developer' as developer;

import 'package:camera/camera.dart';

/// Utilities for converting camera frames between color spaces.
///
/// Handles platform-specific camera image formats:
/// - Android: YUV420 (NV21 or YV12) -> RGB
/// - iOS: BGRA8888 -> RGB
///
/// Optimized for minimal allocations per frame. The output is a flat
/// Uint8List of [width * height * 3] bytes in RGB order, suitable for
/// feeding into the TFLite model after normalization.
class ImageConverter {
  ImageConverter._();

  /// Converts a [CameraImage] to a flat RGB byte array.
  ///
  /// Returns null if the image format is unsupported or conversion fails.
  /// The returned bytes are in row-major RGB order: [R, G, B, R, G, B, ...].
  static Uint8List? convertCameraImageToRgb(CameraImage image) {
    try {
      switch (image.format.group) {
        case ImageFormatGroup.yuv420:
          return _convertYuv420ToRgb(image);
        case ImageFormatGroup.bgra8888:
          return _convertBgraToRgb(image);
        default:
          _log('warn', 'Unsupported image format: ${image.format.group}');
          return null;
      }
    } catch (e) {
      _log('error', 'Image conversion failed: $e');
      return null;
    }
  }

  /// Returns the dimensions of the camera image as (width, height).
  static (int, int) getImageDimensions(CameraImage image) {
    return (image.width, image.height);
  }

  /// Converts YUV420 (Android) to RGB.
  ///
  /// YUV420 layout:
  /// - Plane 0: Y (luminance), full resolution
  /// - Plane 1: U (Cb), half resolution
  /// - Plane 2: V (Cr), half resolution
  ///
  /// On Android, planes may have different row strides and pixel strides
  /// depending on the device. This implementation handles both NV21-style
  /// interleaved UV and planar YV12 formats.
  static Uint8List _convertYuv420ToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final rgbBytes = Uint8List(width * height * 3);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    int rgbIndex = 0;

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        final yIndex = row * yRowStride + col;
        final uvRow = row >> 1;
        final uvCol = col >> 1;
        final uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;

        // Clamp indices to avoid out-of-bounds on edge cases.
        final y = yIndex < yBytes.length ? yBytes[yIndex] : 0;
        final u = uvIndex < uBytes.length ? uBytes[uvIndex] : 128;
        final v = uvIndex < vBytes.length ? vBytes[uvIndex] : 128;

        // YUV to RGB conversion (BT.601 standard).
        int r = (y + 1.370705 * (v - 128)).round();
        int g = (y - 0.337633 * (u - 128) - 0.698001 * (v - 128)).round();
        int b = (y + 1.732446 * (u - 128)).round();

        rgbBytes[rgbIndex++] = r.clamp(0, 255);
        rgbBytes[rgbIndex++] = g.clamp(0, 255);
        rgbBytes[rgbIndex++] = b.clamp(0, 255);
      }
    }

    return rgbBytes;
  }

  /// Converts BGRA8888 (iOS) to RGB.
  ///
  /// BGRA layout is a single plane with 4 bytes per pixel: [B, G, R, A].
  /// We simply extract the R, G, B channels and discard alpha.
  static Uint8List _convertBgraToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final rgbBytes = Uint8List(width * height * 3);

    final plane = image.planes[0];
    final bgraBytes = plane.bytes;
    final rowStride = plane.bytesPerRow;

    int rgbIndex = 0;

    for (int row = 0; row < height; row++) {
      int bgraIndex = row * rowStride;
      for (int col = 0; col < width; col++) {
        // BGRA order: bytes are [B, G, R, A].
        rgbBytes[rgbIndex++] = bgraBytes[bgraIndex + 2]; // R
        rgbBytes[rgbIndex++] = bgraBytes[bgraIndex + 1]; // G
        rgbBytes[rgbIndex++] = bgraBytes[bgraIndex];     // B
        bgraIndex += 4;
      }
    }

    return rgbBytes;
  }

  /// Resizes RGB bytes to the target dimensions using nearest-neighbor
  /// interpolation.
  ///
  /// This is intentionally simple and fast. Bilinear interpolation would
  /// produce slightly better quality but at a meaningful CPU cost per frame.
  /// For model input at 640x640 from 1080p, nearest-neighbor is acceptable.
  static Uint8List resizeRgb(
    Uint8List rgbBytes, {
    required int srcWidth,
    required int srcHeight,
    required int dstWidth,
    required int dstHeight,
  }) {
    final output = Uint8List(dstWidth * dstHeight * 3);

    final xRatio = srcWidth / dstWidth;
    final yRatio = srcHeight / dstHeight;

    int outIndex = 0;
    for (int y = 0; y < dstHeight; y++) {
      final srcY = (y * yRatio).toInt();
      for (int x = 0; x < dstWidth; x++) {
        final srcX = (x * xRatio).toInt();
        final srcIndex = (srcY * srcWidth + srcX) * 3;

        output[outIndex++] = rgbBytes[srcIndex];
        output[outIndex++] = rgbBytes[srcIndex + 1];
        output[outIndex++] = rgbBytes[srcIndex + 2];
      }
    }

    return output;
  }

  /// Converts RGB byte array to Float32List normalized to [0.0, 1.0].
  ///
  /// This is the final step before feeding into the TFLite model.
  /// Output shape is [width * height * 3] as float32 values.
  static Float32List rgbToFloat32(Uint8List rgbBytes) {
    final float32 = Float32List(rgbBytes.length);
    for (int i = 0; i < rgbBytes.length; i++) {
      float32[i] = rgbBytes[i] / 255.0;
    }
    return float32;
  }

  static void _log(String level, String message) {
    developer.log(
      message,
      name: 'ImageConverter',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }
}
