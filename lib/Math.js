/**
 * The lerp() function stands for linear interpolation,
 * and it calculates a value at a specific percentage
 * or ratio between a starting point and an ending point.
 *
 * @param {*} A First Point
 * @param {*} B Second Point
 * @param {*} t Percentage between points [0;1]
 * @returns
 */
function lerp(A, B, t) {
  return A + (B - A) * t;
}

/**
 * Finds the point where two line segments intersect.
 *
 * It solves both segment equations using the 2D cross product. `t` describes
 * how far along AB the intersection is, while `u` describes how far along CD
 * it is. When both values are between 0 and 1, the intersection lies within
 * both finite segments. The returned `offset` is `t`, which callers can use
 * to determine the closest intersection along AB.
 *
 * @param {{ x: number, y: number }} A Start of the first segment
 * @param {{ x: number, y: number }} B End of the first segment
 * @param {{ x: number, y: number }} C Start of the second segment
 * @param {{ x: number, y: number }} D End of the second segment
 * @returns {{ x: number, y: number, offset: number } | null}
 */
function getIntersection(A, B, C, D) {
  const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
  const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
  const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);

  if (bottom != 0) {
    const t = tTop / bottom;
    const u = uTop / bottom;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      return {
        x: lerp(A.x, B.x, t),
        y: lerp(A.y, B.y, t),
        offset: t,
      };
    }
  }

  return null;
}
