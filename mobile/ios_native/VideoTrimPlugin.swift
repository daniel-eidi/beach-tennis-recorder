import Flutter
import AVFoundation

/// Native iOS video trimming plugin using AVFoundation.
///
/// Registered as a MethodChannel handler for "com.beachtennis.recorder/video_trim".
/// Uses AVAssetExportSession with passthrough preset for fast, lossless trimming.
/// Falls back to medium quality re-encoding if passthrough fails.
///
/// [AGENT-01] TASK-01-12: Native video trimming for highlight sharing.
class VideoTrimPlugin: NSObject, FlutterPlugin {

    static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "com.beachtennis.recorder/video_trim",
            binaryMessenger: registrar.messenger()
        )
        let instance = VideoTrimPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "trimVideo":
            guard let args = call.arguments as? [String: Any],
                  let inputPath = args["inputPath"] as? String,
                  let outputPath = args["outputPath"] as? String,
                  let startMs = args["startMs"] as? Int,
                  let endMs = args["endMs"] as? Int else {
                result(FlutterError(
                    code: "INVALID_ARGS",
                    message: "Missing or invalid arguments. Required: inputPath, outputPath, startMs, endMs",
                    details: nil
                ))
                return
            }
            trimVideo(
                inputPath: inputPath,
                outputPath: outputPath,
                startMs: startMs,
                endMs: endMs,
                result: result
            )
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func trimVideo(
        inputPath: String,
        outputPath: String,
        startMs: Int,
        endMs: Int,
        result: @escaping FlutterResult
    ) {
        let inputURL = URL(fileURLWithPath: inputPath)
        let outputURL = URL(fileURLWithPath: outputPath)

        // Remove output file if it already exists.
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outputPath) {
            try? fileManager.removeItem(at: outputURL)
        }

        let asset = AVURLAsset(url: inputURL, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])

        // Convert milliseconds to CMTime.
        let startTime = CMTime(value: CMTimeValue(startMs), timescale: 1000)
        let endTime = CMTime(value: CMTimeValue(endMs), timescale: 1000)

        // Clamp endTime to asset duration.
        let duration = asset.duration
        let clampedEnd = CMTimeMinimum(endTime, duration)
        let clampedStart = CMTimeMaximum(startTime, CMTime.zero)

        guard CMTimeCompare(clampedStart, clampedEnd) < 0 else {
            result(FlutterError(
                code: "INVALID_RANGE",
                message: "Start time must be less than end time. start=\(startMs)ms end=\(endMs)ms duration=\(CMTimeGetSeconds(duration))s",
                details: nil
            ))
            return
        }

        let timeRange = CMTimeRange(start: clampedStart, end: clampedEnd)

        // Try passthrough first (fastest, no re-encoding).
        exportWithPreset(
            asset: asset,
            timeRange: timeRange,
            outputURL: outputURL,
            preset: AVAssetExportPresetPassthrough
        ) { [weak self] success, error in
            if success {
                result(outputPath)
            } else {
                // Fallback: re-encode with medium quality.
                NSLog("[VideoTrimPlugin] Passthrough failed: \(error?.localizedDescription ?? "unknown"). Falling back to MediumQuality.")

                // Clean up failed output.
                try? fileManager.removeItem(at: outputURL)

                self?.exportWithPreset(
                    asset: asset,
                    timeRange: timeRange,
                    outputURL: outputURL,
                    preset: AVAssetExportPresetMediumQuality
                ) { fallbackSuccess, fallbackError in
                    if fallbackSuccess {
                        result(outputPath)
                    } else {
                        result(FlutterError(
                            code: "TRIM_FAILED",
                            message: "Video trim failed: \(fallbackError?.localizedDescription ?? "unknown error")",
                            details: nil
                        ))
                    }
                }
            }
        }
    }

    private func exportWithPreset(
        asset: AVURLAsset,
        timeRange: CMTimeRange,
        outputURL: URL,
        preset: String,
        completion: @escaping (Bool, Error?) -> Void
    ) {
        guard let exportSession = AVAssetExportSession(asset: asset, presetName: preset) else {
            completion(false, NSError(
                domain: "VideoTrimPlugin",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Could not create export session with preset \(preset)"]
            ))
            return
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mp4
        exportSession.timeRange = timeRange
        exportSession.shouldOptimizeForNetworkUse = true

        exportSession.exportAsynchronously {
            DispatchQueue.main.async {
                switch exportSession.status {
                case .completed:
                    completion(true, nil)
                case .failed:
                    completion(false, exportSession.error)
                case .cancelled:
                    completion(false, NSError(
                        domain: "VideoTrimPlugin",
                        code: -2,
                        userInfo: [NSLocalizedDescriptionKey: "Export was cancelled"]
                    ))
                default:
                    completion(false, NSError(
                        domain: "VideoTrimPlugin",
                        code: -3,
                        userInfo: [NSLocalizedDescriptionKey: "Export ended with unexpected status: \(exportSession.status.rawValue)"]
                    ))
                }
            }
        }
    }
}
