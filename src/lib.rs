//!
//! Provides some useful 2d geometry functions.
//!
//! Why the name? Not sure. Duck Duck Goose.

extern crate axgeom;


pub mod bot;

use axgeom::Vec2;
use axgeom::vec2;
use axgeom::ordered_float::NotNan;
use axgeom::Rect;
use axgeom::num_traits::Float;
use axgeom::num_traits::Num;
use axgeom::num_traits::Zero;
use core::ops::Neg;
use axgeom::num_traits::NumAssign;


pub trait MyNum: Zero+Copy+NumAssign + PartialOrd + Neg<Output=Self>{
}
impl<T:Zero+Copy+NumAssign+PartialOrd+Neg<Output=Self>> MyNum for T{}
//use cgmath::num_traits::NumCast;

pub type F64n=NotNan<f64>;
pub type F32n=NotNan<f32>;



pub fn array2_inner_into<B:Copy,A:From<B>>(a:[B;2])->[A;2]{
    let x=A::from(a[0]);
    let y=A::from(a[1]);
    [x,y]
}

use core::convert::TryFrom;

pub fn array2_inner_try_into<B:Copy,A:TryFrom<B>>(a:[B;2])->Result<[A;2],A::Error>{
    let x=A::try_from(a[0]);
    let y=A::try_from(a[1]);
    match (x,y){
        (Ok(x),Ok(y))=>{
            Ok([x,y])
        },
        (Ok(_),Err(e))=>{
            Err(e)
        },
        (Err(e),Ok(_))=>{
            Err(e)
        },
        (Err(e1),Err(_))=>{
            Err(e1)
        }
    }
}


use core::ops::Sub;
use core::ops::Add;
pub fn rect_from_point<N:Copy+Sub<Output=N>+Add<Output=N>>(point:Vec2<N>,radius:Vec2<N>)->Rect<N>{
    Rect::new(point.x-radius.x,point.x+radius.x,point.y-radius.y,point.y+radius.y)
}



///Passed to gravitate.
pub trait GravityTrait {
    type N: Float;
    fn pos(&self) -> Vec2<Self::N>;
    fn mass(&self) -> Self::N;
    fn apply_force(&mut self, arr: Vec2<Self::N>);
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
    type N: Float;
    fn pos(&self) -> Vec2<Self::N>;
    fn add_force(&mut self, force: Vec2<Self::N>);
}

///If we repel too close, because of the inverse square we might get overlow problems.
pub struct ErrTooClose;

///Repel one object by simply not calling add_force on the other.
pub fn repel_one<B: RepelTrait>(
    bot1: &mut B,
    pos: Vec2<B::N>,
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
    type N: MyNum;
    fn pos_vel_mut(&mut self) -> (&mut Vec2<Self::N>,&mut Vec2<Self::N>);
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


pub fn stop_wall<N:MyNum>(pos: &mut Vec2<N>, dim: Vec2<N>){
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
pub fn wrap_position<N:MyNum>(a: &mut Vec2<N>, dim: Rect<N>) {
    let ((a_,b),(c,d))=dim.get();

    let start = vec2(a_,c);
    let dim=vec2(b,d);

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
pub fn collide_with_rect<N: Float>(
    botr: &axgeom::Rect<N>,
    wallr: &axgeom::Rect<N>,
) -> Option<WallSide> {
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

    let ans=match [d1.partial_cmp(&zero)?, d2.partial_cmp(&zero)?] {
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
    };
    Some(ans)
}



///A Ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray<N> {
    pub point: Vec2<N>,
    pub dir: Vec2<N>,
}

impl<N> Ray<N> {
    pub fn new(point:Vec2<N>,dir:Vec2<N>)->Ray<N>{
        Ray{point,dir}
    }

    pub fn inner_into<B:From<N>>(self)->Ray<B>{
        let point=self.point.inner_into();
        let dir=self.dir.inner_into();
        Ray{point,dir}
    }
    pub fn inner_try_into<B:TryFrom<N>>(self)->Result<Ray<B>,B::Error>{
        let point=self.point.inner_try_into();
        let dir=self.dir.inner_try_into();
        match(point,dir){
            (Ok(point),Ok(dir))=>{
                Ok(Ray{point,dir})
            },
            (Err(e),Ok(_))=>{
                Err(e)
            },
            (Ok(_),Err(e))=>{
                Err(e)
            },
            (Err(e),Err(_))=>{
                Err(e)
            }
        }
    }
}




//Given a ray and an axis aligned line, return the tvalue,and x coordinate
pub fn ray_compute_intersection_tvalue<A: axgeom::AxisTrait,N:MyNum>(
    ray:&Ray<N>,
    axis: A,
    line: N,
) -> Option<(N)> {
    

    if axis.is_xaxis(){
        if ray.dir.x == N::zero() {
            if ray.point.x == line {
                Some(N::zero())
            } else {
                None
            }
        } else {
            let t = (line - ray.point.x) / ray.dir.x;
            if t >= N::zero()
            {
                Some(t)
            } else {
                None
            }
        }
    }else{
        if ray.dir.y == N::zero() {
            if ray.point.y == line {
                Some(N::zero())
            } else {
                None
            }
        } else {
            let t = (line - ray.point.y) / ray.dir.y;
            if t >= N::zero()
            {
                Some(t)
            } else {
                None
            }
        }
    }
    
}



//We are making these generic for floats. Integers and floats behave fundamentally different.
//Dont make generic math functions over both. Example sqrt() behaves differently.
pub fn ray_intersects_box_int<N:MyNum+Ord>(ray:&Ray<N>, rect: &axgeom::Rect<N>) -> IntersectsBotResult<N>{
    let point = ray.point;
    let dir = ray.dir;

    let ((x1, x2), (y1, y2)) = rect.get();

    //val=t*m+y
    let (tmin, tlen) = if dir.x != N::zero() {
        let tx1 = (x1 - point.x) / dir.x;
        let tx2 = (x2 - point.x) / dir.x;

        (tx1.min(tx2), tx1.max(tx2))
    } else if point.x < x1 || point.x > x2 {
        return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
    } else {
        return IntersectsBotResult::Hit(N::zero()); //TODO i think this is wrong?
    };

    let (tmin, tlen) = if dir.y != N::zero() {
        let ty1 = (y1 - point.y) / dir.y;
        let ty2 = (y2 - point.y) / dir.y;

        let k1=tmin.max(ty1.min(ty2));
        let k2=tlen.min(ty1.max(ty2));
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





///Returns if a ray intersects a box.
pub fn ray_intersects_box<N:Float>(ray:&Ray<N>, rect: &axgeom::Rect<N>) -> IntersectsBotResult<N> {
    let point = ray.point;
    let dir = ray.dir;

    let ((x1, x2), (y1, y2)) = rect.get();

    //val=t*m+y
    let (tmin, tlen) = if dir.x != N::zero() {
        let tx1 = (x1 - point.x) / dir.x;
        let tx2 = (x2 - point.x) / dir.x;

        (tx1.min(tx2), tx1.max(tx2))
    } else if point.x < x1 || point.x > x2 {
        return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
    } else {
        return IntersectsBotResult::Hit(N::zero()); //TODO i think this is wrong?
    };

    let (tmin, tlen) = if dir.y != N::zero() {
        let ty1 = (y1 - point.y) / dir.y;
        let ty2 = (y2 - point.y) / dir.y;

        let k1=tmin.max(ty1.min(ty2));
        let k2=tlen.min(ty1.max(ty2));
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


///Describes if a ray hit a rectangle.
#[derive(Copy, Clone, Debug)]
pub enum IntersectsBotResult<N> {
    Hit(N),
    Inside,
    NoHit,
}
