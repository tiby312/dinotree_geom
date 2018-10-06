extern crate num_traits;
extern crate axgeom;
extern crate num;

use num_traits::Num;


pub trait GravityTrait{
    type N:Num+PartialOrd+Copy;
    fn pos(&self)->[Self::N;2];
    fn mass(&self)->Self::N;
    fn apply_force(&mut self,[Self::N;2]);
}

//Returns the force to be exerted to the first object.
//The force to the second object can be retrieved simply by negating the first.
pub fn gravitate<N:Num+PartialOrd+Copy,T:GravityTrait<N=N>,T2:GravityTrait<N=N>>(a:&mut T,b:&mut T2,min:N,gravity_const:N,sqrt:impl Fn(N)->N)->Result<(),ErrTooClose>{
    let p1=a.pos();
    let p2=b.pos();
    let m1=a.mass();
    let m2=b.mass();

    let diffx=p2[0]-p1[0];
    let diffy=p2[1]-p1[1];
    let dis_sqr=diffx*diffx+diffy*diffy;


    if dis_sqr>min{
        
        //const GRAVITY_CONSTANT:f64=0.004;

        //newtons law of gravitation (modified for 2d??? divide by len instead of sqr)
        let force=gravity_const*(m1*m2)/dis_sqr;

        let dis=sqrt(dis_sqr);
        let finalx=diffx*(force/dis);
        let finaly=diffy*(force/dis);
        
        a.apply_force([finalx,finaly]);
        b.apply_force([N::zero()-finalx,N::zero()-finaly]);
        Ok(())
    }else{
        Err(ErrTooClose)
    }
}

#[derive(Debug,Copy,Clone)]
pub struct Ray<N>{
    pub point:[N;2],
    pub dir:[N;2],
    pub tlen:N,
}


fn sub<N:Num+Copy>(a:[N;2],b:[N;2])->[N;2]{
    [b[0]-a[0],b[1]-a[1]]
}
fn derive_center<N:Num+Copy>(a:&axgeom::Rect<N>)->[N;2]{
    let two=N::one()+N::one();
    let ((a,b),(c,d))=a.get();
    [a+(b-a)/two,c+(d-c)/two]
}

fn dot<N:Num+Copy>(a:[N;2],b:[N;2])->N{
   a[0]*b[0]+a[1]*b[1] 
}


pub trait RepelTrait{
    type N:Num+Copy+PartialOrd;
    fn pos(&self)->[Self::N;2];
    fn add_force(&mut self,force:[Self::N;2]);
}

pub struct ErrTooClose;
pub fn repel_one<B:RepelTrait>(bot1:&mut B,pos:[B::N;2],closest:B::N,mag:B::N,sqrt:impl Fn(B::N)->B::N)->Result<(),ErrTooClose>{
    let a=bot1;

    let pos1=a.pos();
    let pos2=pos;
    let diff=[pos2[0]-pos1[0],pos2[1]-pos1[1]];

    let len_sqr=diff[0]*diff[0]+diff[1]*diff[1];

    if len_sqr<closest{
        return Err(ErrTooClose)
    }

    let len=sqrt(len_sqr);
    let mag=mag/len;

    let norm=[diff[0]/len,diff[1]/len];
    
    let zero=<B::N as num_traits::Zero>::zero();
    a.add_force([zero-norm[0]*mag,zero-norm[1]*mag]);
    //b.add_force([norm[0]*mag,norm[1]*mag]);
    
    return Ok(())
}
pub fn repel<B:RepelTrait>(bot1:&mut B,bot2:&mut B,closest:B::N,mag:B::N,sqrt:impl Fn(B::N)->B::N)->Result<(),ErrTooClose>{
    let a=bot1;
    let b=bot2;

    let pos1=a.pos();
    let pos2=b.pos();
    let diff=[pos2[0]-pos1[0],pos2[1]-pos1[1]];

    let len_sqr=diff[0]*diff[0]+diff[1]*diff[1];

    if len_sqr<closest{
        return Err(ErrTooClose)
    }

    let len=sqrt(len_sqr);
    let mag=mag/len;

    let norm=[diff[0]/len,diff[1]/len];
    
    let zero=<B::N as num_traits::Zero>::zero();
    a.add_force([zero-norm[0]*mag,zero-norm[1]*mag]);
    b.add_force([norm[0]*mag,norm[1]*mag]);
    
    return Ok(())
}
    


pub fn wrap_position<N:Num+Copy+PartialOrd>(a:&mut [N;2],dim:[N;2]){
    let start=[N::zero();2];

    if a[0]>dim[0]{
        a[0]=start[0]
    }
    if a[0]<start[0]{
        a[0]=dim[0];
    }
    if a[1]>dim[1]{
        a[1]=start[1];
    }
    if a[1]<start[1]{
        a[1]=dim[1];
    }
}


pub enum WallSide{
    Above,
    Below,
    LeftOf,
    RightOf
}

pub fn collide_with_rect<N:Num+Copy+Ord>(botr:&axgeom::Rect<N>,wallr:&axgeom::Rect<N>)->WallSide{

    let wallx=wallr.get_range(axgeom::XAXISS);
    let wally=wallr.get_range(axgeom::YAXISS);

    
    let center_bot=derive_center(botr);
    let center_wall=derive_center(wallr);

    let ratio=(wallx.right-wallx.left)/(wally.right-wally.left);

    //Assuming perfect square
    let p1=[N::one(),ratio];
    let p2=[N::zero()-N::one(),ratio];


    let diff=sub(center_wall,center_bot);

    let d1=dot(p1,diff);
    let d2=dot(p2,diff);
    let zero=N::zero();

    use std::cmp::Ordering::*;

    
    match [d1.cmp(&zero),d2.cmp(&zero)]{
        [Less,Less]=>{
            //top
            WallSide::Above
        }
        [Less,Equal]=>{
            //topleft
            WallSide::Above
        },
        [Less,Greater]=>{
            //left
            WallSide::LeftOf
        },
        [Greater,Less]=>{
            //right
            WallSide::RightOf
        },
        [Greater,Equal]=>{
            //bottom right
            WallSide::Below
        },
        [Greater,Greater]=>{
            //bottom
            WallSide::Below
        },
        [Equal,Less]=>{
            //top right
            WallSide::Above
        },
        [Equal,Equal]=>{
            //Directly in the center
            WallSide::Above
        },
        [Equal,Greater]=>{
            //bottom left
            WallSide::Below
        }
    }
}


pub fn distance_squred_point<N:Num+Copy+PartialOrd>(point1:[N;2],point2:[N;2])->N{
    let x=point2[0]-point1[0];
    let y=point2[1]-point1[1];
    x*x+y*y
}

///Returns the squred distance from a point to a rectangle if the point is outisde the rectangle.
///If the point is insert the rectangle, it will return None.
pub fn distance_squared_point_to_rect<N:Num+Copy+PartialOrd>(point:[N;2],rect:&axgeom::Rect<N>)->Option<N>{
    let (px,py)=(point[0],point[1]);

    let ((a,b),(c,d))=rect.get();

    let xx=num::clamp(px,a,b);
    let yy=num::clamp(py,c,d);

    
    let dis=(xx-px)*(xx-px) + (yy-py)*(yy-py);

    //Then the point must be insert the rect.
    //In this case, lets return something negative.
    if xx>a && xx<b && yy>c && yy< d{
        None
    }else{
        Some(dis)
    }
}


/*
pub fn split_ray<N:Num+Copy+Ord,A:axgeom::AxisTrait>(axis:A,ray:&Ray<N>,fo:N)->Option<(Ray<N>,Ray<N>)>{
    let two=N::one()+N::one();

    let t=if axis.is_xaxis(){
        if ray.dir[0]==N::zero(){
            if ray.point[0]==fo{
                let t1=ray.tlen/two;
                let t2=ray.tlen-t1;
                //Lets just split it into half.
                let ray_closer=Ray{point:ray.point,dir:ray.dir,tlen:t1};
                let new_point=[ray.point[0],ray.point[1]+t1];
                let ray_new=Ray{point:new_point,dir:ray.dir,tlen:t2};

                return Some((ray_closer,ray_new))
            }else{
                return None
            }
        }else{
            (fo-ray.point[0])/ray.dir[0]
        }
    }else{
        if ray.dir[1]==N::zero(){
            if ray.point[1]==fo{
                let t1=ray.tlen/two;
                let t2=ray.tlen-t1;
                //Lets just split it into half.
                let ray_closer=Ray{point:ray.point,dir:ray.dir,tlen:t1};
                let new_point=[ray.point[0]+t1,ray.point[1]];
                let ray_new=Ray{point:new_point,dir:ray.dir,tlen:t2};

                return Some((ray_closer,ray_new))
            }else{
                return None
            }
        }else{
            (fo-ray.point[1])/ray.dir[1]   
        }
    };

    if t>ray.tlen || t<N::zero(){
        return None
    }

    let new_point=[ray.point[0]+ray.dir[0]*t,ray.point[1]+ray.dir[1]*t];
    
    let ray_closer=Ray{point:ray.point,dir:ray.dir,tlen:t};
    let ray_new=Ray{point:new_point,dir:ray.dir,tlen:ray.tlen-t};
    Some((ray_closer,ray_new))
    
}
*/

/*
///Returns a lower and upper bound around the intersection y value.
pub fn compute_intersection_range<N:Num+Copy+Ord,A:axgeom::AxisTrait>(ray:&Ray<N>,axis:A,line:N)->Option<N>
{
    let o1=compute_intersection_tvalue(axis,&ray,line).map(|tvalue|{
        convert_tvalue_to_point(axis.next(),&ray,tvalue)
    });
    
    match o1{
        Some(dis)=>{
            let [ray_origin_x,ray_origin_y,ray_end_y]=if axis.is_xaxis(){
                [ray.point[0],ray.point[1],ray.point[1]+ray.tlen*ray.dir[1]]
            }else{
                [ray.point[1],ray.point[0],ray.point[0]+ray.tlen*ray.dir[0]]
            };
            Some(ray_end_y)
        },
        None=>{
            None
        }
    }
}
*/

/*
//First option is min, second is max
pub fn compute_intersection_range<N:Num+Copy+Ord,A:axgeom::AxisTrait>(ray:&Ray<N>,axis:A,fat_line:[N;2])->Option<(N,N)>
{
    let o1=compute_intersection_tvalue(axis,&ray,fat_line[0]).map(|tvalue|{
        convert_tvalue_to_point(axis.next(),&ray,tvalue)
    });
    let o2=compute_intersection_tvalue(axis,&ray,fat_line[1]).map(|tvalue|{
        convert_tvalue_to_point(axis.next(),&ray,tvalue)
    });
 
    let [ray_origin_x,ray_origin_y,ray_end_y]=if axis.is_xaxis(){
        [ray.point[0],ray.point[1],ray.point[1]+ray.tlen*ray.dir[1]]
    }else{
        [ray.point[1],ray.point[0],ray.point[0]+ray.tlen*ray.dir[0]]
    };

    let origin_inside=ray_origin_x>=fat_line[0] && ray_origin_x<=fat_line[1];

    match (o1,o2){
        (Some(a),None)=>{ 
            if origin_inside{
                Some((a.min(ray_origin_y),a.max(ray_origin_y)))
            }else{
                Some((a.min(ray_end_y),a.max(ray_end_y)))
            }
        },
        (None,Some(a))=>{
            if origin_inside{
                Some((a.min(ray_origin_y),a.max(ray_origin_y)))
            }else{
                Some((a.min(ray_end_y),a.max(ray_end_y)))
            }
        },
        (Some(a),Some(b))=>{
            Some((a.min(b),b.max(a)))
        },
        (None,None)=>{
            //TODO figure out inequalities
            if origin_inside{
                Some((ray_origin_y.min(ray_end_y),ray_origin_y.max(ray_end_y)))
            }else{
                None
            }
        }
    }
}
*/
  

pub fn convert_tvalue_to_point<N:Num+Copy+Ord,A:axgeom::AxisTrait>(axis:A,ray:&Ray<N>,tvalue:N)->N{
    //y=mx+b
    if axis.is_xaxis(){
        ray.dir[0]*tvalue+ray.point[0]
    }else{
        ray.dir[1]*tvalue+ray.point[1]
    }
}


//Given a ray and an axis aligned line, return the tvalue,and x coordinate
pub fn compute_intersection_tvalue<N:Num+Copy+Ord,A:axgeom::AxisTrait>(axis:A,ray:&Ray<N>,line:N)->Option<(N)>{
    if axis.is_xaxis(){
        if ray.dir[0]==N::zero(){
            if ray.point[0]==line{
                Some(N::zero())
            }else{
                None
            }
        }else{
            let t=(line-ray.point[0])/ray.dir[0];
            
            if t>=N::zero() && t<=ray.tlen{
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
            if t>=N::zero() && t<=ray.tlen{
                Some(t)
            }else{
                None
            }
        }
    }
}

#[derive(Copy,Clone,Debug)]
pub enum IntersectsBotResult<N>{
    Hit(N),
    Inside,
    NoHit
}


pub fn intersects_box<N:Num+Copy+Ord>(point:[N;2],dir:[N;2],matt:N,rect:&axgeom::Rect<N>)->IntersectsBotResult<N>{
    let ((x1,x2),(y1,y2))=rect.get();

    //val=t*m+y
    let (tmin,tlen)=if dir[0]!=N::zero(){
        let tx1=(x1-point[0])/dir[0];
        let tx2=(x2-point[0])/dir[0];

        (tx1.min(tx2),
        tx1.max(tx2))
    }else{
        if point[0] < x1 || point[0] > x2 {
            return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
        }else{
            (N::zero(),matt)
        }
    };

    let (tmin,tlen)=if dir[1]!=N::zero(){
        let ty1=(y1-point[1])/dir[1];
        let ty2=(y2-point[1])/dir[1];

        (tmin.max(ty1.min(ty2)),
        tlen.min(ty1.max(ty2)))
    }else{
        if point[1] < y1 || point[1] > y2 {
            return IntersectsBotResult::NoHit; // parallel AND outside box : no intersection possible
        }else{
            (tmin,tlen)
        }
    };

    //TODO figure out inequalities!
    if tmin<=N::zero() && tlen>=N::zero(){
        return IntersectsBotResult::Inside;
    }

    if tmin<=N::zero() && tlen<N::zero(){
        return IntersectsBotResult::NoHit;
    }

    if tmin>matt{
        return IntersectsBotResult::NoHit;
    }
    

    if tlen>=tmin{
        return IntersectsBotResult::Hit(tmin);
    }else{
        return IntersectsBotResult::NoHit;
    }
                
}

