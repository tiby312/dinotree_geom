//!
//! Provides some useful 2d geometry functions.
//!
//! Why the name? Not sure. Duck Duck Goose.

extern crate axgeom;
extern crate num;
extern crate num_traits;

use num_traits::Num;

pub mod vec2f32;
pub mod vec2f64;
pub mod bot;

///Passed to gravitate.
pub trait GravityTrait {
    type N: Num + PartialOrd + Copy;
    fn pos(&self) -> [Self::N; 2];
    fn mass(&self) -> Self::N;
    fn apply_force(&mut self, arr: [Self::N; 2]);
}

///Returns the force to be exerted to the first object.
///The force to the second object can be retrieved simply by negating the first.
pub fn gravitate<N: Num + PartialOrd + Copy, T: GravityTrait<N = N>, T2: GravityTrait<N = N>>(
    a: &mut T,
    b: &mut T2,
    min: N,
    gravity_const: N,
    sqrt: impl Fn(N) -> N,
) -> Result<(), ErrTooClose> {
    let p1 = a.pos();
    let p2 = b.pos();
    let m1 = a.mass();
    let m2 = b.mass();

    let diffx = p2[0] - p1[0];
    let diffy = p2[1] - p1[1];
    let dis_sqr = diffx * diffx + diffy * diffy;

    if dis_sqr > min {
        //newtons law of gravitation (modified for 2d??? divide by len instead of sqr)
        let force = gravity_const * (m1 * m2) / dis_sqr;

        let dis = sqrt(dis_sqr);
        let finalx = diffx * (force / dis);
        let finaly = diffy * (force / dis);

        a.apply_force([finalx, finaly]);
        b.apply_force([N::zero() - finalx, N::zero() - finaly]);
        Ok(())
    } else {
        Err(ErrTooClose)
    }
}

fn sub<N: Num + Copy>(a: [N; 2], b: [N; 2]) -> [N; 2] {
    [b[0] - a[0], b[1] - a[1]]
}
fn derive_center<N: Num + Copy>(a: &axgeom::Rect<N>) -> [N; 2] {
    let two = N::one() + N::one();
    let ((a, b), (c, d)) = a.get();
    [a + (b - a) / two, c + (d - c) / two]
}

fn dot<N: Num + Copy>(a: [N; 2], b: [N; 2]) -> N {
    a[0] * b[0] + a[1] * b[1]
}

///Passed to repel
pub trait RepelTrait {
    type N: Num + Copy + PartialOrd;
    fn pos(&self) -> [Self::N; 2];
    fn add_force(&mut self, force: [Self::N; 2]);
}

///If we repel too close, because of the inverse square we might get overlow problems.
pub struct ErrTooClose;

///Repel one object by simply not calling add_force on the other.
pub fn repel_one<B: RepelTrait>(
    bot1: &mut B,
    pos: [B::N; 2],
    closest: B::N,
    mag: B::N,
    sqrt: impl Fn(B::N) -> B::N,
) -> Result<(), ErrTooClose> {
    let a = bot1;

    let pos1 = a.pos();
    let pos2 = pos;
    let diff = [pos2[0] - pos1[0], pos2[1] - pos1[1]];

    let len_sqr = diff[0] * diff[0] + diff[1] * diff[1];

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = sqrt(len_sqr);
    let mag = mag / len;

    let norm = [diff[0] / len, diff[1] / len];

    let zero = <B::N as num_traits::Zero>::zero();
    a.add_force([zero - norm[0] * mag, zero - norm[1] * mag]);

    Ok(())
}

///Repel two objects.
pub fn repel<B: RepelTrait>(
    bot1: &mut B,
    bot2: &mut B,
    closest: B::N,
    mag: B::N,
    sqrt: impl Fn(B::N) -> B::N,
) -> Result<(), ErrTooClose> {
    let a = bot1;
    let b = bot2;

    let pos1 = a.pos();
    let pos2 = b.pos();
    let diff = [pos2[0] - pos1[0], pos2[1] - pos1[1]];

    let len_sqr = diff[0] * diff[0] + diff[1] * diff[1];

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = sqrt(len_sqr);
    let mag = mag / len;

    let norm = [diff[0] / len, diff[1] / len];

    let zero = <B::N as num_traits::Zero>::zero();
    a.add_force([zero - norm[0] * mag, zero - norm[1] * mag]);
    b.add_force([norm[0] * mag, norm[1] * mag]);

    Ok(())
}


pub fn stop_wall<N:Num+Copy+PartialOrd>(pos:&mut [N;2],dim:[N;2]){
    let start = [N::zero();2];

    if pos[0] > dim[0]{
        pos[0] = dim[0];
    }
    if pos[0] < start[0]{
        pos[0]=start[0];
    }


    if pos[1] > dim[1]{
        pos[1] = dim[1];
    }
    if pos[1] < start[1]{
        pos[1]=start[1];
    }
}


///Wraps the first point around the rectangle made between (0,0) and dim.
pub fn wrap_position<N: Num + Copy + PartialOrd>(a: &mut [N; 2], dim: [N; 2]) {
    let start = [N::zero(); 2];

    if a[0] > dim[0] {
        a[0] = start[0]
    }
    if a[0] < start[0] {
        a[0] = dim[0];
    }
    if a[1] > dim[1] {
        a[1] = start[1];
    }
    if a[1] < start[1] {
        a[1] = dim[1];
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
pub fn collide_with_rect<N: Num + Copy + Ord>(
    botr: &axgeom::Rect<N>,
    wallr: &axgeom::Rect<N>,
) -> WallSide {
    let wallx = wallr.get_range(axgeom::XAXISS);
    let wally = wallr.get_range(axgeom::YAXISS);

    let center_bot = derive_center(botr);
    let center_wall = derive_center(wallr);

    let ratio = (wallx.right - wallx.left) / (wally.right - wally.left);

    //Assuming perfect square
    let p1 = [N::one(), ratio];
    let p2 = [N::zero() - N::one(), ratio];

    let diff = sub(center_wall, center_bot);

    let d1 = dot(p1, diff);
    let d2 = dot(p2, diff);
    let zero = N::zero();

    use std::cmp::Ordering::*;

    match [d1.cmp(&zero), d2.cmp(&zero)] {
        [Less, Less] => {
            //top
            WallSide::Above
        }
        [Less, Equal] => {
            //topleft
            WallSide::Above
        }
        [Less, Greater] => {
            //left
            WallSide::LeftOf
        }
        [Greater, Less] => {
            //right
            WallSide::RightOf
        }
        [Greater, Equal] => {
            //bottom right
            WallSide::Below
        }
        [Greater, Greater] => {
            //bottom
            WallSide::Below
        }
        [Equal, Less] => {
            //top right
            WallSide::Above
        }
        [Equal, Equal] => {
            //Directly in the center
            WallSide::Above
        }
        [Equal, Greater] => {
            //bottom left
            WallSide::Below
        }
    }
}

///Returns the squared distances between two points.
pub fn distance_squred_point<N: Num + Copy + PartialOrd>(point1: [N; 2], point2: [N; 2]) -> N {
    let x = point2[0] - point1[0];
    let y = point2[1] - point1[1];
    x * x + y * y
}

///If the point is outisde the rectangle, returns the squared distance from a point to a rectangle.
///If the point is insert the rectangle, it will return None.
pub fn distance_squared_point_to_rect<N: Num + Copy + PartialOrd>(
    point: [N; 2],
    rect: &axgeom::Rect<N>,
) -> Option<N> {
    let (px, py) = (point[0], point[1]);

    let ((a, b), (c, d)) = rect.get();

    let xx = num::clamp(px, a, b);
    let yy = num::clamp(py, c, d);

    let dis = (xx - px) * (xx - px) + (yy - py) * (yy - py);

    //Then the point must be insert the rect.
    //In this case, lets return something negative.
    if xx > a && xx < b && yy > c && yy < d {
        None
    } else {
        Some(dis)
    }
}

///A Ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray<N> {
    pub point: [N; 2],
    pub dir: [N; 2],
}

impl<N: Num + Copy + Ord> Ray<N> {
    //Given a ray and an axis aligned line, return the tvalue,and x coordinate
    pub fn compute_intersection_tvalue<A: axgeom::AxisTrait>(
        &self,
        axis: A,
        line: N,
    ) -> Option<(N)> {
        let ray = self;
        /*
        if axis.is_xaxis(){
            if ray.dir[0]==N::zero(){
                if ray.point[0]==line{
                    Some(N::zero())
                }else{
                    None
                }
            }else{
                let t=(line-ray.point[0])/ray.dir[0];

                if t>=N::zero() /*&& t<=ray.tlen*/{
                    Some(t)
                }else{
                    None
                }
            }
        }else{
            if ray.dir[1]==N::zero(){
                if ray.point[1]==line{
                    Some(N::zero())
                }else{
                    None
                }
            }else{

                let t=(line-ray.point[1])/ray.dir[1];
                if t>=N::zero() /*&& t<=ray.tlen*/{
                    Some(t)
                }else{
                    None
                }
            }
        }
        */
        let axis = if axis.is_xaxis() { 0 } else { 1 };
        if ray.dir[axis] == N::zero() {
            if ray.point[axis] == line {
                Some(N::zero())
            } else {
                None
            }
        } else {
            let t = (line - ray.point[axis]) / ray.dir[axis];
            if t >= N::zero()
            /*&& t<=ray.tlen*/
            {
                Some(t)
            } else {
                None
            }
        }
    }
    ///Returns if a ray intersects a box.
    pub fn intersects_box(&self, rect: &axgeom::Rect<N>) -> IntersectsBotResult<N> {
        let point = self.point;
        let dir = self.dir;
        let ((x1, x2), (y1, y2)) = rect.get();

        //val=t*m+y
        let (tmin, tlen) = if dir[0] != N::zero() {
            let tx1 = (x1 - point[0]) / dir[0];
            let tx2 = (x2 - point[0]) / dir[0];

            (tx1.min(tx2), tx1.max(tx2))
        } else if point[0] < x1 || point[0] > x2 {
            return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
        } else {
            return IntersectsBotResult::Hit(N::zero()); //TODO i think this is wrong?
        };

        let (tmin, tlen) = if dir[1] != N::zero() {
            let ty1 = (y1 - point[1]) / dir[1];
            let ty2 = (y2 - point[1]) / dir[1];

            (tmin.max(ty1.min(ty2)), tlen.min(ty1.max(ty2)))
        } else if point[1] < y1 || point[1] > y2 {
            return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
        } else {
            (tmin, tlen)
        };

        //TODO figure out inequalities!
        if tmin <= N::zero() && tlen >= N::zero() {
            return IntersectsBotResult::Inside;
        }

        if tmin <= N::zero() && tlen < N::zero() {
            return IntersectsBotResult::NoHit;
        }

        if tlen >= tmin {
            IntersectsBotResult::Hit(tmin)
        } else {
            IntersectsBotResult::NoHit
        }
    }
}

///Describes if a ray hit a rectangle.
#[derive(Copy, Clone, Debug)]
pub enum IntersectsBotResult<N> {
    Hit(N),
    Inside,
    NoHit,
}
