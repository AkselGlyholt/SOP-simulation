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
