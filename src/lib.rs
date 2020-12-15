//!
//! Provides some useful 2d geometry functions.
//!
#[macro_use]
extern crate more_asserts;

extern crate axgeom;

pub use dists;

///2d grid library with the ability to the raycast to detect
///which cell a ray hits.
pub mod grid;

use axgeom::num_traits::Float;
use axgeom::num_traits::NumAssign;
use axgeom::num_traits::Zero;
use axgeom::vec2;
use axgeom::Rect;
use axgeom::Vec2;
use core::ops::Neg;

///Basic Number Trait
pub trait MyNum: Zero + Copy + NumAssign + PartialOrd + Neg<Output = Self> {}
impl<T: Zero + Copy + NumAssign + PartialOrd + Neg<Output = Self>> MyNum for T {}

///convert an array of elements of type B to type A.
pub fn array2_inner_into<B: Copy, A: From<B>>(a: [B; 2]) -> [A; 2] {
    let x = A::from(a[0]);
    let y = A::from(a[1]);
    [x, y]
}

use core::convert::TryFrom;

///convert an array of elements of type B to type A.
pub fn array2_inner_try_into<B: Copy, A: TryFrom<B>>(a: [B; 2]) -> Result<[A; 2], A::Error> {
    let x = A::try_from(a[0]);
    let y = A::try_from(a[1]);
    match (x, y) {
        (Ok(x), Ok(y)) => Ok([x, y]),
        (Ok(_), Err(e)) => Err(e),
        (Err(e), Ok(_)) => Err(e),
        (Err(e1), Err(_)) => Err(e1),
    }
}

///Returns the force to be exerted to the first object.
///The force to the second object can be retrieved simply by negating the first.
pub fn gravitate<N: Float + MyNum>(
    bots: [(Vec2<N>, N, &mut Vec2<N>); 2],
    min: N,
    gravity_const: N,
) -> Result<(), ErrTooClose> {
    let [(p1, m1, f1), (p2, m2, f2)] = bots;

    let diff = p2 - p1;
    let dis_sqr = diff.magnitude2();

    if dis_sqr > min {
        //newtons law of gravitation (modified for 2d??? divide by len instead of sqr)
        let force = gravity_const * (m1 * m2) / dis_sqr;

        let dis = Float::sqrt(dis_sqr);

        let final_vec = diff * (force / dis);

        *f1 += final_vec;
        *f2 -= final_vec;
        Ok(())
    } else {
        Err(ErrTooClose)
    }
}

///If we repel too close, because of the inverse square we might get overlow problems.
pub struct ErrTooClose;

///Repel one object by simply not calling add_force on the other.
pub fn repel_one<N: Float + MyNum>(
    pos1: Vec2<N>,
    force_buffer: &mut Vec2<N>,
    pos2: Vec2<N>,
    closest: N,
    mag: N,
) -> Result<(), ErrTooClose> {
    let diff = pos2 - pos1;

    let len_sqr = diff.magnitude2();

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = Float::sqrt(len_sqr);
    let mag = mag / len;

    let force = diff.normalize_to(mag);

    *force_buffer -= force;

    Ok(())
}

pub fn linear_push<N: Float + MyNum>(
    bots: [(Vec2<N>, &mut Vec2<N>); 2],
    closest: N,
    mag: N,
) -> Result<(), ErrTooClose> {
    let [(bot1_pos, bot1_force_buffer), (bot2_pos, bot2_force_buffer)] = bots;

    let diff = bot2_pos - bot1_pos;

    let len_sqr = diff.magnitude2();

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = len_sqr.sqrt();
    let mag = mag / len;

    let force = diff.normalize_to(mag);

    *bot1_force_buffer -= force;
    *bot2_force_buffer += force;

    Ok(())
}

///Repel two objects.
///First vector is position. Second vector is force buffer
pub fn repel<N: Float + MyNum>(
    bots: [(Vec2<N>, &mut Vec2<N>); 2],
    closest: N,
    mag: N,
) -> Result<(), ErrTooClose> {
    let [(bot1_pos, bot1_force_buffer), (bot2_pos, bot2_force_buffer)] = bots;

    let diff = bot2_pos - bot1_pos;

    let len_sqr = diff.magnitude2();

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = len_sqr.sqrt();
    let mag = mag / len;

    let force = diff.normalize_to(mag);

    *bot1_force_buffer -= force;
    *bot2_force_buffer += force;

    Ok(())
}

///Collides and bounces an object with a border
pub fn collide_with_border<N: MyNum>(
    pos: &mut Vec2<N>,
    vel: &mut Vec2<N>,
    rect2: &axgeom::Rect<N>,
    drag: N,
) {
    let xx = rect2.get_range(axgeom::XAXIS);
    let yy = rect2.get_range(axgeom::YAXIS);

    if pos.x < xx.start {
        pos.x = xx.start;
        vel.x = -vel.x;
        vel.x *= drag;
    }
    if pos.x > xx.end {
        pos.x = xx.end;
        vel.x = -vel.x;
        vel.x *= drag;
    }
    if pos.y < yy.start {
        pos.y = yy.start;
        vel.y = -vel.y;
        vel.y *= drag;
    }
    if pos.y > yy.end {
        pos.y = yy.end;
        vel.y = -vel.y;
        vel.y *= drag;
    }
}

///Forces a position to be within the specified rect.
#[inline(always)]
pub fn stop_wall<N: MyNum>(pos: &mut Vec2<N>, rect: Rect<N>) {
    let start = vec2(rect.x.start, rect.y.start);
    let dim = vec2(rect.x.end, rect.y.end);
    if pos.x > dim.x {
        pos.x = dim.x;
    } else if pos.x < start.x {
        pos.x = start.x;
    }

    if pos.y > dim.y {
        pos.y = dim.y;
    } else if pos.y < start.y {
        pos.y = start.y;
    }
}

///Wraps the first point around the rectangle made between (0,0) and dim.
pub fn wrap_position<N: MyNum>(a: &mut Vec2<N>, dim: Rect<N>) {
    let ((a_, b), (c, d)) = dim.get();

    let start = vec2(a_, c);
    let dim = vec2(b, d);

    if a.x > dim.x {
        a.x = start.x
    } else if a.x < start.x {
        a.x = dim.x;
    }

    if a.y > dim.y {
        a.y = start.y;
    } else if a.y < start.y {
        a.y = dim.y;
    }
}

///Describes a cardinal direction..
pub enum WallSide {
    Above,
    Below,
    LeftOf,
    RightOf,
}

///Returns which cardinal direction the specified rectangle is closest to.
pub fn collide_with_rect<N: Float>(
    botr: &axgeom::Rect<N>,
    wallr: &axgeom::Rect<N>,
) -> Option<WallSide> {
    let wallx = wallr.get_range(axgeom::XAXIS);
    let wally = wallr.get_range(axgeom::YAXIS);

    let center_bot = botr.derive_center();
    let center_wall = wallr.derive_center();

    let ratio = (wallx.end - wallx.start) / (wally.end - wally.start);

    //Assuming perfect square
    let p1 = vec2(N::one(), ratio);
    let p2 = vec2(N::zero() - N::one(), ratio);

    let diff = center_bot - center_wall;

    let d1 = p1.dot(diff);
    let d2 = p2.dot(diff);
    let zero = N::zero();

    use std::cmp::Ordering::*;

    let ans = match [d1.partial_cmp(&zero)?, d2.partial_cmp(&zero)?] {
        [Less, Less] => {
            //top
            WallSide::Above
        }
        [Less, Equal] => {
            //topstart
            WallSide::Above
        }
        [Less, Greater] => {
            //start
            WallSide::LeftOf
        }
        [Greater, Less] => {
            //end
            WallSide::RightOf
        }
        [Greater, Equal] => {
            //bottom end
            WallSide::Below
        }
        [Greater, Greater] => {
            //bottom
            WallSide::Below
        }
        [Equal, Less] => {
            //top end
            WallSide::Above
        }
        [Equal, Equal] => {
            //Directly in the center
            WallSide::Above
        }
        [Equal, Greater] => {
            //bottom start
            WallSide::Below
        }
    };
    Some(ans)
}
