/// Represents a single object detection from the YOLOv8 TFLite model.
///
/// Coordinates are in pixel space relative to the input image dimensions.
/// The model outputs [x, y, w, h, confidence, classId] per detection.
class Detection {
  /// Center X coordinate of the bounding box (pixels).
  final double x;

  /// Center Y coordinate of the bounding box (pixels).
  final double y;

  /// Width of the bounding box (pixels).
  final double w;

  /// Height of the bounding box (pixels).
  final double h;

  /// Confidence score from the model (0.0 - 1.0).
  final double confidence;

  /// Numeric class identifier from the model.
  final int classId;

  /// Human-readable class name.
  final String className;

  const Detection({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.confidence,
    required this.classId,
    required this.className,
  });

  /// Top-left X of the bounding box.
  double get left => x - w / 2;

  /// Top-left Y of the bounding box.
  double get top => y - h / 2;

  /// Bottom-right X of the bounding box.
  double get right => x + w / 2;

  /// Bottom-right Y of the bounding box.
  double get bottom => y + h / 2;

  /// Area of the bounding box in pixels squared.
  double get area => w * h;

  /// Whether this detection represents a ball.
  bool get isBall => classId == DetectionClass.ball;

  /// Whether this detection represents a net.
  bool get isNet => classId == DetectionClass.net;

  /// Whether this detection represents a court line.
  bool get isCourtLine => classId == DetectionClass.courtLine;

  /// Whether this detection represents a player.
  bool get isPlayer => classId == DetectionClass.player;

  /// Whether this detection represents a racket.
  bool get isRacket => classId == DetectionClass.racket;

  /// Creates a Detection from the raw model output array.
  ///
  /// [raw] is expected to be [x, y, w, h, confidence, classId].
  /// [classNames] maps classId to className.
  factory Detection.fromModelOutput(
    List<double> raw, {
    Map<int, String> classNames = DetectionClass.names,
  }) {
    final classId = raw[5].toInt();
    return Detection(
      x: raw[0],
      y: raw[1],
      w: raw[2],
      h: raw[3],
      confidence: raw[4],
      classId: classId,
      className: classNames[classId] ?? 'unknown',
    );
  }

  Map<String, dynamic> toJson() => {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'confidence': confidence,
        'classId': classId,
        'className': className,
      };

  factory Detection.fromJson(Map<String, dynamic> json) => Detection(
        x: (json['x'] as num).toDouble(),
        y: (json['y'] as num).toDouble(),
        w: (json['w'] as num).toDouble(),
        h: (json['h'] as num).toDouble(),
        confidence: (json['confidence'] as num).toDouble(),
        classId: json['classId'] as int,
        className: json['className'] as String,
      );

  @override
  String toString() =>
      'Detection($className @ ($x,$y) ${w}x$h conf=$confidence)';
}

/// Class IDs matching the trained YOLOv8 model output.
///
/// Must stay in sync with vision/dataset/data.yaml classes.
abstract class DetectionClass {
  static const int ball = 0;
  static const int net = 1;
  static const int courtLine = 2;
  static const int player = 3;
  static const int racket = 4;

  static const Map<int, String> names = {
    ball: 'ball',
    net: 'net',
    courtLine: 'court_line',
    player: 'player',
    racket: 'racket',
  };
}
