import 'dart:math' as math;

/// Standard beach tennis court dimensions in meters.
///
/// Official ITF beach tennis court:
///   Length: 16m (including service areas)
///   Width: 8m
///   Net height: 1.7m
///   Net position: center (8m from each baseline)
abstract class CourtDimensions {
  static const double lengthMeters = 16.0;
  static const double widthMeters = 8.0;
  static const double netPositionMeters = 8.0; // from each baseline
  static const double netHeightMeters = 1.7;

  /// Standard court corners in real-world coordinates (meters).
  /// Origin is at top-left corner of the court.
  static const List<Point2D> standardCorners = [
    Point2D(0, 0),                         // top-left
    Point2D(widthMeters, 0),               // top-right
    Point2D(widthMeters, lengthMeters),    // bottom-right
    Point2D(0, lengthMeters),              // bottom-left
  ];
}

/// A 2D point with double coordinates.
class Point2D {
  final double x;
  final double y;

  const Point2D(this.x, this.y);

  double distanceTo(Point2D other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  Map<String, dynamic> toJson() => {'x': x, 'y': y};

  factory Point2D.fromJson(Map<String, dynamic> json) => Point2D(
        (json['x'] as num).toDouble(),
        (json['y'] as num).toDouble(),
      );

  @override
  String toString() => 'Point2D($x, $y)';

  @override
  bool operator ==(Object other) =>
      other is Point2D && x == other.x && y == other.y;

  @override
  int get hashCode => Object.hash(x, y);
}

/// Represents a court calibration: 4 screen-space corners mapped to a
/// standard beach tennis court.
///
/// Computes a 3x3 homography matrix that transforms screen coordinates to
/// real-world court coordinates (in meters). This enables:
///   - Accurate out-of-bounds detection
///   - Net zone identification
///   - Distance calculations on the court
///
/// Calibration is done ONCE per location and saved locally as JSON.
class CourtCalibration {
  /// Unique identifier for this calibration.
  final String id;

  /// User-given name for this location (e.g., "Ipanema Court 3").
  final String name;

  /// The 4 screen-space corners tapped by the user, in order:
  ///   [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left.
  final List<Point2D> screenCorners;

  /// The computed 3x3 homography matrix (row-major, 9 elements).
  /// Transforms screen coordinates -> court coordinates.
  final List<double> homographyMatrix;

  /// The inverse homography matrix.
  /// Transforms court coordinates -> screen coordinates.
  final List<double> inverseHomographyMatrix;

  /// When this calibration was created.
  final DateTime createdAt;

  const CourtCalibration({
    required this.id,
    required this.name,
    required this.screenCorners,
    required this.homographyMatrix,
    required this.inverseHomographyMatrix,
    required this.createdAt,
  });

  /// Creates a calibration from 4 screen-space corner taps.
  ///
  /// [corners] must contain exactly 4 points in order:
  ///   top-left, top-right, bottom-right, bottom-left.
  /// [name] is a user-friendly label for this location.
  factory CourtCalibration.fromCorners({
    required String id,
    required String name,
    required List<Point2D> corners,
  }) {
    assert(corners.length == 4, 'Exactly 4 corners required');

    final homography = _computeHomography(
      corners,
      CourtDimensions.standardCorners,
    );
    final inverse = _computeHomography(
      CourtDimensions.standardCorners,
      corners,
    );

    return CourtCalibration(
      id: id,
      name: name,
      screenCorners: List.unmodifiable(corners),
      homographyMatrix: homography,
      inverseHomographyMatrix: inverse,
      createdAt: DateTime.now(),
    );
  }

  /// Transforms a screen-space point to court-space (meters).
  ///
  /// Returns the position on the standard court plane.
  Point2D transformPoint(Point2D screenPoint) {
    return _applyHomography(homographyMatrix, screenPoint);
  }

  /// Transforms a court-space point (meters) to screen-space.
  Point2D inverseTransformPoint(Point2D courtPoint) {
    return _applyHomography(inverseHomographyMatrix, courtPoint);
  }

  /// Whether a court-space point is inside the court boundaries.
  bool isInsideCourt(Point2D courtPoint) {
    return courtPoint.x >= 0 &&
        courtPoint.x <= CourtDimensions.widthMeters &&
        courtPoint.y >= 0 &&
        courtPoint.y <= CourtDimensions.lengthMeters;
  }

  /// Whether a screen-space point maps to outside the court.
  bool isOutOfBounds(Point2D screenPoint) {
    final courtPoint = transformPoint(screenPoint);
    return !isInsideCourt(courtPoint);
  }

  /// Returns the net zone as a rectangle in court-space (meters).
  ///
  /// The net zone is a narrow strip across the center of the court,
  /// used for detecting net crossings.
  ({Point2D topLeft, Point2D bottomRight}) getNetZone() {
    const halfWidth = 0.5; // 0.5m on each side of the net line
    return (
      topLeft: Point2D(0, CourtDimensions.netPositionMeters - halfWidth),
      bottomRight: Point2D(
        CourtDimensions.widthMeters,
        CourtDimensions.netPositionMeters + halfWidth,
      ),
    );
  }

  /// Whether a court-space point is in the net zone.
  bool isInNetZone(Point2D courtPoint) {
    final zone = getNetZone();
    return courtPoint.x >= zone.topLeft.x &&
        courtPoint.x <= zone.bottomRight.x &&
        courtPoint.y >= zone.topLeft.y &&
        courtPoint.y <= zone.bottomRight.y;
  }

  /// Serializes this calibration to JSON for disk persistence.
  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'screenCorners': screenCorners.map((p) => p.toJson()).toList(),
        'homographyMatrix': homographyMatrix,
        'inverseHomographyMatrix': inverseHomographyMatrix,
        'createdAt': createdAt.toIso8601String(),
      };

  /// Deserializes a calibration from JSON.
  factory CourtCalibration.fromJson(Map<String, dynamic> json) {
    return CourtCalibration(
      id: json['id'] as String,
      name: json['name'] as String,
      screenCorners: (json['screenCorners'] as List)
          .map((p) => Point2D.fromJson(p as Map<String, dynamic>))
          .toList(),
      homographyMatrix: (json['homographyMatrix'] as List)
          .map((v) => (v as num).toDouble())
          .toList(),
      inverseHomographyMatrix: (json['inverseHomographyMatrix'] as List)
          .map((v) => (v as num).toDouble())
          .toList(),
      createdAt: DateTime.parse(json['createdAt'] as String),
    );
  }

  @override
  String toString() => 'CourtCalibration("$name" id=$id)';

  // ---------------------------------------------------------------------------
  // Homography math
  // ---------------------------------------------------------------------------

  /// Applies a 3x3 homography matrix to a 2D point.
  ///
  /// The homography maps (x, y) -> (x', y') via:
  ///   [h0 h1 h2]   [x]   [wx']
  ///   [h3 h4 h5] * [y] = [wy']
  ///   [h6 h7 h8]   [1]   [w  ]
  ///   x' = wx'/w,  y' = wy'/w
  static Point2D _applyHomography(List<double> h, Point2D p) {
    final w = h[6] * p.x + h[7] * p.y + h[8];
    if (w.abs() < 1e-10) return const Point2D(0, 0);
    final x = (h[0] * p.x + h[1] * p.y + h[2]) / w;
    final y = (h[3] * p.x + h[4] * p.y + h[5]) / w;
    return Point2D(x, y);
  }

  /// Computes a 3x3 homography matrix from 4 source points to 4 destination
  /// points using the Direct Linear Transform (DLT) algorithm.
  ///
  /// Given 4 point correspondences (src_i -> dst_i), solves the 8-equation
  /// system for the 8 degrees of freedom of the homography (h8 = 1).
  ///
  /// This is a pure Dart implementation — no OpenCV or native dependencies.
  static List<double> _computeHomography(
    List<Point2D> src,
    List<Point2D> dst,
  ) {
    assert(src.length == 4 && dst.length == 4);

    // Build the 8x8 matrix A and 8x1 vector b for Ah = b.
    // For each correspondence (x,y) -> (x',y'):
    //   x' = (h0*x + h1*y + h2) / (h6*x + h7*y + 1)
    //   y' = (h3*x + h4*y + h5) / (h6*x + h7*y + 1)
    // Rearranged:
    //   h0*x + h1*y + h2 - h6*x*x' - h7*y*x' = x'
    //   h3*x + h4*y + h5 - h6*x*y' - h7*y*y' = y'

    final a = List.generate(8, (_) => List.filled(8, 0.0));
    final b = List.filled(8, 0.0);

    for (int i = 0; i < 4; i++) {
      final sx = src[i].x;
      final sy = src[i].y;
      final dx = dst[i].x;
      final dy = dst[i].y;

      final row1 = i * 2;
      final row2 = i * 2 + 1;

      // Row for x' equation.
      a[row1][0] = sx;
      a[row1][1] = sy;
      a[row1][2] = 1;
      a[row1][3] = 0;
      a[row1][4] = 0;
      a[row1][5] = 0;
      a[row1][6] = -sx * dx;
      a[row1][7] = -sy * dx;
      b[row1] = dx;

      // Row for y' equation.
      a[row2][0] = 0;
      a[row2][1] = 0;
      a[row2][2] = 0;
      a[row2][3] = sx;
      a[row2][4] = sy;
      a[row2][5] = 1;
      a[row2][6] = -sx * dy;
      a[row2][7] = -sy * dy;
      b[row2] = dy;
    }

    // Solve 8x8 linear system using Gaussian elimination with partial pivoting.
    final h8 = _solveLinearSystem(a, b);

    // h8 = 1 (normalization).
    return [...h8, 1.0];
  }

  /// Solves Ax = b using Gaussian elimination with partial pivoting.
  ///
  /// [a] is an NxN matrix, [b] is an Nx1 vector. Returns x.
  static List<double> _solveLinearSystem(
    List<List<double>> a,
    List<double> b,
  ) {
    final n = b.length;

    // Augment the matrix [A|b].
    final aug = List.generate(
      n,
      (i) => [...a[i], b[i]],
    );

    // Forward elimination with partial pivoting.
    for (int col = 0; col < n; col++) {
      // Find pivot.
      int maxRow = col;
      double maxVal = aug[col][col].abs();
      for (int row = col + 1; row < n; row++) {
        final val = aug[row][col].abs();
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }

      // Swap rows.
      if (maxRow != col) {
        final temp = aug[col];
        aug[col] = aug[maxRow];
        aug[maxRow] = temp;
      }

      final pivot = aug[col][col];
      if (pivot.abs() < 1e-12) continue; // Singular or near-singular.

      // Eliminate below.
      for (int row = col + 1; row < n; row++) {
        final factor = aug[row][col] / pivot;
        for (int j = col; j <= n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }

    // Back substitution.
    final x = List.filled(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
      double sum = aug[i][n];
      for (int j = i + 1; j < n; j++) {
        sum -= aug[i][j] * x[j];
      }
      final diag = aug[i][i];
      x[i] = diag.abs() < 1e-12 ? 0.0 : sum / diag;
    }

    return x;
  }
}
