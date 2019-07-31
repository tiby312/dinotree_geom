

use axgeom::Rect;
use ordered_float::*;
use dists;
use cgmath::prelude::*;
use cgmath::Vector2;
use cgmath::vec2;



pub struct BotScene{
    pub bot_prop:BotProp,
    pub bots:Vec<Bot>
}

pub struct BotSceneBuilder{
    grow:f64,
    radius:f64,
    num:usize,
    bot_radius:f32
}
impl BotSceneBuilder{
    pub fn new(num:usize)->BotSceneBuilder{
        BotSceneBuilder{grow:0.2,radius:17.0,num,bot_radius:5.0}
    }

    pub fn with_grow(&mut self,grow:f64)->&mut Self{
        self.grow=grow;
        self
    }
    pub fn with_num(&mut self,num:usize)->&mut Self{
        self.num=num;
        self
    }

    pub fn with_radius_of(&mut self,radius:f64)->&mut Self{
        self.radius=radius;
        self
    }

    pub fn build(&mut self)->BotScene{
        
        let spiral=dists::spiral::Spiral::new([0.0,0.0],self.radius,self.grow).as_f32();
        


        let bots:Vec<Bot>=spiral.take(self.num).map(|pos|{
            Bot::new(vec2(pos[0],pos[1]))
        }).collect();

        let bot_prop=BotProp{
            radius:Dist::new(self.bot_radius),
            collision_push:0.1,
            collision_drag:0.1,
            minimum_dis_sqr:0.0001,
            viscousity_coeff:0.1};

        BotScene{bot_prop,bots}
    }
}









#[derive(Copy,Clone,Debug)]
pub struct BotProp {
    pub radius: Dist,
    pub collision_push: f32,
    pub collision_drag: f32,
    pub minimum_dis_sqr: f32,
    pub viscousity_coeff: f32
}




impl BotProp{


    pub fn collide(&self,a:&mut Bot,b:&mut Bot){

        //Takes a values between 0 and 1, and returns a value between 0 and 1.
        //The input is the distance from not touching.
        //So if the bots are just barely touching, the input will be 0.
        //If they are right on top of each other, the input will be 1.

        //The output is the normalized force with which to handle.
        //A value of 0 is no force.
        //A value of 1 is full force.
        pub fn handle_repel(input:f32)->f32{
            let a=3.0*input*input;
            a.min(1.0)
        }

        
        let prop=self;
        let bots=[a,b];

        let offset = bots[0].pos - bots[1].pos;

        let dis_sqr = offset.magnitude2();
        
        if dis_sqr >= prop.radius.dis2_squared() {
            //They not touching (bots are circular).
            return ;
        }

        //At this point, we know they collide!!!!

        let sum_rad = prop.radius.dis2();

        let dis = dis_sqr.sqrt();

        
        //0 if barely touching (not touching)
        //1 if directly on top of each other
        let dd=(sum_rad-dis)/sum_rad;

        let ammount_touching=handle_repel(dd);

        let push_mag= ammount_touching*prop.collision_push;
        
        let velocity_diff=bots[0].vel-bots[1].vel;

        let drag=-prop.collision_drag*ammount_touching*velocity_diff.dot(offset);
            
        let push1=drag+push_mag;
        let push2=-drag-push_mag;

        let push_force1=offset*(push1/dis);
        let push_force2=offset*(push2/dis);

        let viscous=velocity_diff*-prop.viscousity_coeff*ammount_touching;

        bots[0].acc+=push_force1;
        bots[0].acc+=viscous;

        bots[1].acc+=push_force2;
        bots[1].acc+=viscous;
    }






    pub fn collide_mouse(&self,bot:&mut Bot,mouse:&Mouse){
        let prop=self;
        let offset = *mouse.get_midpoint() - bot.pos;
        let dis_sqr = offset.magnitude2();
        
        let sum_rad=mouse.get_radius() + prop.radius.dis();
        if dis_sqr < sum_rad*sum_rad {

            let dis = dis_sqr.sqrt();

            if dis<0.0001{
                return;
            }

            let vv=(sum_rad-dis)/sum_rad;
            let vv=vv*vv;
            let vv2=(5.0*vv).min(1.0);


            let push_mag=vv2*mouse.mouse_prop.force;
            let push_force=offset*(push_mag/dis);

            bot.acc+=-push_force;
        }
    }

}



#[derive(Copy,Clone,Debug)]
pub struct Bot{
    pub pos: Vector2<f32>,
    pub vel: Vector2<f32>,
    pub acc: Vector2<f32>,
}

impl crate::BorderCollideTrait for Bot{
    type N=f32;
    fn pos_vel_mut(&mut self)->(&mut Vector2<f32>,&mut Vector2<f32>){
        (&mut self.pos,&mut self.vel)
    }
}
impl Bot{
    pub fn apply(&mut self,bot:&Bot){
        self.acc=bot.acc;
    }

    pub fn create_bbox(&self,bot_scene:&BotProp)->Rect<f32>{
        let p=self.pos;
        let r=bot_scene.radius.dis();
        Rect::new(p.x-r,p.x+r,p.y-r,p.y+r)
    }

    pub fn create_bbox_nan(&self,bot_scene:&BotProp)->Rect<NotNan<f32>>{
        self.create_bbox(bot_scene).into_notnan().unwrap()
    }

    pub fn new(pos:Vector2<f32>)->Bot{
        let vel=Vector2::zero();
        let acc=Vector2::zero();
        Bot{pos,vel,acc}
    }

    #[inline]
    pub fn pos(&self)->&Vector2<f32>{
        &self.pos
    }

    #[inline]
    pub fn vel(&self)->&Vector2<f32>{
        &self.vel
    }

    pub fn push_away(&mut self,b:&mut Self,radius:f32,max_amount:f32){
        let mut diff=b.pos-self.pos;

        let dis=diff.magnitude();

        if dis<0.000001{
            return;
        }

        let mag=max_amount.min(radius*2.0-dis);
        if mag<0.0{
            return;
        }
        //let mag=max_amount;
        diff*=mag/dis;

        self.acc-=diff;
        b.acc+=diff;

        //TODO if we have moved too far away, move back to point of collision!!!
        {

        }
    }
}



#[derive(Copy,Clone,Debug)]
pub struct MouseProp {
    pub radius: Dist,
    pub force: f32,
}



#[derive(Copy,Clone,Debug)]
pub struct Mouse{
    pub mouse_prop: MouseProp,
    pub midpoint:Vector2<f32>,
    pub rect:axgeom::Rect<f32>
}
impl Mouse{
    pub fn new(pos:Vector2<f32>,prop:&MouseProp)->Mouse{
        let mut m:Mouse=unsafe{std::mem::uninitialized()};
        m.mouse_prop= *prop;
        m.move_to(pos);
        m
    }

    pub fn get_rect(&self)->&axgeom::Rect<f32>{
        &self.rect
    }
    pub fn get_midpoint(&self)->&Vector2<f32>{
        &self.midpoint
    }
    pub fn get_radius(&self)->f32{
        self.mouse_prop.radius.dis()
    }
    pub fn move_to(&mut self,pos:Vector2<f32>){
        self.midpoint= pos;
        let p=self.midpoint;
        let r=self.mouse_prop.radius.dis();
        let r=axgeom::Rect::new(p.x-r,p.x+r,p.y-r,p.y+r);
        self.rect=r;
    }
}



///A struct with cached calculations based off of a radius.
#[derive(Copy,Clone,Debug)]
pub struct Dist {
    dis: f32,
    dis2: f32,
    dis2_squared: f32,
}
impl Dist {

    #[inline]
    pub fn new(dis: f32) -> Dist {
        let k = dis * 2.0;

        Dist {
            dis,
            dis2: k,
            dis2_squared: k.powi(2),
            //radius_x_root_2_inv: radius * (1.0 / 1.4142),
        }
    }

    ///Returns the rdius
    #[inline]
    pub fn dis(&self) -> f32 {
        self.dis
    }
    
    ///Returns the cached radius*2.0
    #[inline]
    pub fn dis2(&self) -> f32 {
        self.dis2
    }
    
    ///Returns the cached radius.powi(2)
    #[inline]
    pub fn dis2_squared(&self) -> f32 {
        self.dis2_squared
    }
}


use rand;
use rand::{SeedableRng, StdRng};
use rand::distributions::{IndependentSample, Range};



pub struct RangeGenIterf64{
    max:usize,
    counter:usize,
    rng:rand::StdRng,
    xvaluegen:UniformRangeGenerator,
    yvaluegen:UniformRangeGenerator,
    radiusgen:UniformRangeGenerator,
    velocity_dir:UniformRangeGenerator,
    velocity_mag:UniformRangeGenerator
}

pub struct Retf64{
    pub id:usize,
    pub pos:[f64;2],
    pub vel:[f64;2],
    pub radius:[f64;2],
}

pub struct RetInteger{
    pub id:usize,
    pub pos:[isize;2],
    pub vel:[isize;2],
    pub radius:[isize;2],
}
impl Retf64{
    pub fn into_isize(self)->RetInteger{
        let id=self.id;
        let pos=[self.pos[0] as isize,self.pos[1] as isize];
        let vel=[self.vel[0] as isize,self.vel[1] as isize];
        let radius=[self.radius[0] as isize,self.radius[1] as isize];
        RetInteger{id,pos,vel,radius}
    }
}
impl std::iter::FusedIterator for RangeGenIterf64{}
impl ExactSizeIterator for RangeGenIterf64{}
impl Iterator for RangeGenIterf64{
    type Item=Retf64;
    fn size_hint(&self)->(usize,Option<usize>){
        (self.max,Some(self.max))
    }
    fn next(&mut self)->Option<Self::Item>{  

        if self.counter==self.max{
            return None
        }

        let rng=&mut self.rng;  
        let px=self.xvaluegen.get(rng) as f64;
        let py=self.yvaluegen.get(rng) as f64;
        let rx=self.radiusgen.get(rng) as f64;
        let ry=self.radiusgen.get(rng) as f64;

        let (velx,vely)={
            let vel_dir=self.velocity_dir.get(rng) as f64;
            let vel_dir=vel_dir.to_radians();
            let (mut xval,mut yval)=(vel_dir.cos(),vel_dir.sin());
            let vel_mag=self.velocity_mag.get(rng) as f64;
            xval*=vel_mag;
            yval*=vel_mag;
            (xval,yval)
        };

        let curr=self.counter;
        self.counter+=1;
        let r=Retf64{id:curr,pos:[px,py],vel:[velx,vely],radius:[rx,ry]};
        Some(r)
    }
}
pub fn create_world_generator(num:usize,area:&[isize;4],radius:[isize;2],velocity:[isize;2])->RangeGenIterf64{
    let arr:&[usize]=&[100,42,6];
    let rng =  SeedableRng::from_seed(arr);


    let xvaluegen=UniformRangeGenerator::new(area[0],area[1]);
    let yvaluegen=UniformRangeGenerator::new(area[2],area[3]);
    let radiusgen= UniformRangeGenerator::new(radius[0],radius[1]);


    let velocity_dir=UniformRangeGenerator::new(0,360);
    let velocity_mag= UniformRangeGenerator::new(velocity[0],velocity[1]);

    RangeGenIterf64{max:num,counter:0,rng,xvaluegen,yvaluegen,radiusgen,velocity_dir,velocity_mag}
}



struct UniformRangeGenerator{
    range:Range<isize>
}

impl UniformRangeGenerator{
    pub fn new(a:isize,b:isize)->Self{
        //let rr = a.get_range2::<axgeom::XAXISS>();
        let xdist = rand::distributions::Range::new(a,b);
        UniformRangeGenerator{range:xdist}
    }
    pub fn get(&self,rng:&mut StdRng)->isize{
        self.range.ind_sample(rng)
    }
}
