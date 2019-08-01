//!
//! Provides some useful 2d geometry functions.
//!
//! Why the name? Not sure. Duck Duck Goose.

extern crate axgeom;


pub mod bot;

use cgmath::num_traits::Float;
use cgmath::prelude::*;
use cgmath::Vector2;
use cgmath::BaseFloat;
use cgmath::vec2;
use ordered_float::NotNan;
use axgeom::Rect;

use cgmath::num_traits::NumCast;

pub type F64n=NotNan<f64>;
pub type F32n=NotNan<f32>;



pub fn cast_2array<K:NumCast+Copy,K2:NumCast+Copy>(a:[K;2])->Option<[K2;2]>{
    let x=K2::from(a[0]);
    let y=K2::from(a[1]);

    match (x,y){
        (Some(x),Some(y))=>{
            Some([x,y])
        },
        _=>{
            None
        }
    }
}

pub fn rect_from_point<N:BaseFloat>(point:Vector2<N>,radius:Vector2<N>)->Rect<N>{
    Rect::new(point.x-radius.x,point.x+radius.x,point.y-radius.y,point.y+radius.y)
}
/*
pub fn point_notnan_to_inner(a:Vector2<F64n>)->Vector2<f64>{
    vec2(a.x.into_inner(),a.y.into_inner())
}
pub struct PointNanErr;
pub fn point_inner_to_notnan(a:Vector2<f64>)->Result<Vector2<F64n>,PointNanErr>{

    let x=NotNan::new(a.x);
    let y=NotNan::new(a.y);
    match(x,y){
        (Ok(x),Ok(y))=>{
            Ok(vec2(x,y))
        },
        _=>{
            Err(PointNanErr)
        }
    }
}
*/


///Passed to gravitate.
pub trait GravityTrait {
    type N: BaseFloat;
    fn pos(&self) -> Vector2<Self::N>;
    fn mass(&self) -> Self::N;
    fn apply_force(&mut self, arr: Vector2<Self::N>);
}

///Returns the force to be exerted to the first object.
///The force to the second object can be retrieved simply by negating the first.
pub fn gravitate<T:GravityTrait>(
    a: &mut T,
    b: &mut T,
    min: T::N,
    gravity_const: T::N
) -> Result<(), ErrTooClose> {
    let p1 = a.pos();
    let p2 = b.pos();
    let m1 = a.mass();
    let m2 = b.mass();

    let diff=p2-p1;
    let dis_sqr=diff.magnitude2();

    if dis_sqr > min {
        //newtons law of gravitation (modified for 2d??? divide by len instead of sqr)
        let force = gravity_const * (m1 * m2) / dis_sqr;

        let dis = Float::sqrt(dis_sqr);
        
        let final_vec = diff * (force / dis);


        a.apply_force(final_vec);
        b.apply_force(-final_vec);
        Ok(())
    } else {
        Err(ErrTooClose)
    }
}




///Passed to repel
pub trait RepelTrait {
    type N: BaseFloat +  One+ Copy + PartialOrd;
    fn pos(&self) -> Vector2<Self::N>;
    fn add_force(&mut self, force: Vector2<Self::N>);
}

///If we repel too close, because of the inverse square we might get overlow problems.
pub struct ErrTooClose;

///Repel one object by simply not calling add_force on the other.
pub fn repel_one<B: RepelTrait>(
    bot1: &mut B,
    pos: Vector2<B::N>,
    closest: B::N,
    mag: B::N
) -> Result<(), ErrTooClose> {
    let a = bot1;

    let pos1 = a.pos();
    let pos2 = pos;
    
    let diff=pos2-pos1;
    
    let len_sqr=diff.magnitude2();

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = Float::sqrt(len_sqr);
    let mag = mag / len;


    let force=diff.normalize_to(mag);

    a.add_force(-force);

    Ok(())
}

///Repel two objects.
pub fn repel<B: RepelTrait>(
    bot1: &mut B,
    bot2: &mut B,
    closest: B::N,
    mag: B::N
) -> Result<(), ErrTooClose> {
    let a = bot1;
    let b = bot2;

    let pos1 = a.pos();
    let pos2 = b.pos();
    let diff = pos2-pos1;

    let len_sqr=diff.magnitude2();

    if len_sqr < closest {
        return Err(ErrTooClose);
    }

    let len = len_sqr.sqrt();
    let mag = mag / len;

    let force=diff.normalize_to(mag);

    a.add_force(-force);
    b.add_force(force);

    Ok(())
}



pub trait BorderCollideTrait{
    type N: BaseFloat + Copy + PartialOrd + core::ops::MulAssign + core::ops::Neg<Output=Self::N>;
    fn pos_vel_mut(&mut self) -> (&mut Vector2<Self::N>,&mut Vector2<Self::N>);
}

pub fn collide_with_border<B:BorderCollideTrait>(
        a:&mut B,rect2:&axgeom::Rect<B::N>,drag:B::N){

    let xx=rect2.get_range(axgeom::XAXISS);
    let yy=rect2.get_range(axgeom::YAXISS);


    let (pos,vel)=&mut a.pos_vel_mut();

    if pos.x<xx.left{
        pos.x=xx.left;
        vel.x= -vel.x;
        vel.x*=drag;
    }
    if pos.x>xx.right{
        pos.x=xx.right;
        vel.x= -vel.x;
        vel.x*=drag;
    }
    if pos.y<yy.left{
        pos.y=yy.left;
        vel.y= -vel.y;
        vel.y*=drag;
    }
    if pos.y>yy.right{
        pos.y=yy.right;
        vel.y= -vel.y;
        vel.y*=drag;
    }

}


pub fn stop_wall<N:BaseFloat+Zero+Copy+PartialOrd>(pos: &mut Vector2<N>, dim: Vector2<N>){
    let start = vec2(N::zero(),N::zero());
    if pos.x > dim.x{
        pos.x = dim.x;
    }else if pos.x < start.x{
        pos.x=start.x;
    }


    if pos.y > dim.y{
        pos.y = dim.y;
    }else if pos.y < start.y{
        pos.y=start.y;
    }
}


///Wraps the first point around the rectangle made between (0,0) and dim.
pub fn wrap_position<N: BaseFloat+ Zero + Copy + PartialOrd>(a: &mut Vector2<N>, dim: Vector2<N>) {
    let start = vec2(N::zero(),N::zero());

    if a.x > dim.x {
        a.x = start.x
    }else if a.x < start.x {
        a.x = dim.x;
    }

    if a.y > dim.y {
        a.y = start.y;
    }else if a.y < start.y {
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
pub fn collide_with_rect<N: BaseFloat+ Copy + Ord>(
    botr: &axgeom::Rect<N>,
    wallr: &axgeom::Rect<N>,
) -> WallSide {
    let wallx = wallr.get_range(axgeom::XAXISS);
    let wally = wallr.get_range(axgeom::YAXISS);

    let center_bot = botr.derive_center();
    let center_wall = wallr.derive_center();

    let ratio = (wallx.right - wallx.left) / (wally.right - wally.left);

    //Assuming perfect square
    let p1 = vec2(N::one(), ratio);
    let p2 = vec2(N::zero() - N::one(), ratio);

    let diff = center_bot-center_wall;

    let d1 = p1.dot(diff);
    let d2 = p2.dot(diff);
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



///A Ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray<N> {
    pub point: Vector2<N>,
    pub dir: Vector2<N>,
}

impl<N: BaseFloat + Copy + Ord> Ray<N> {
    pub fn new(point:Vector2<N>,dir:Vector2<N>)->Ray<N>{
        Ray{point,dir}
    }
    //Given a ray and an axis aligned line, return the tvalue,and x coordinate
    pub fn compute_intersection_tvalue<A: axgeom::AxisTrait>(
        &self,
        axis: A,
        line: N,
    ) -> Option<(N)> {
        let ray = self;
        
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
        let (tmin, tlen) = if dir.x != N::zero() {
            let tx1 = (x1 - point.x) / dir.x;
            let tx2 = (x2 - point.x) / dir.x;

            (Float::min(tx1,tx2), Float::max(tx1,tx2))
        } else if point.x < x1 || point.x > x2 {
            return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
        } else {
            return IntersectsBotResult::Hit(N::zero()); //TODO i think this is wrong?
        };

        let (tmin, tlen) = if dir.y != N::zero() {
            let ty1 = (y1 - point.y) / dir.y;
            let ty2 = (y2 - point.y) / dir.y;

            let k1=Float::max(tmin,Float::min(ty1,ty2));
            let k2=Float::min(tlen,Float::max(ty1,ty2));
            (k1,k2)
        } else if point.y < y1 || point.y > y2 {
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
