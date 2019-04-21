

use axgeom::Rect;
use crate::vec2f32::Vec2;
use ordered_float::*;
use dists;
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
        //radius= 5.0,5.0
        //let s=dists::spiral::Spiral::new([400.0,400.0],17.0,0.2);
          
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
        


        let mut bots:Vec<Bot>=spiral.take(self.num).map(|pos|{
            Bot::new(Vec2::new(pos[0],pos[1]))
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

        let dis_sqr = offset.dis_sqr();
        
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

        let drag=-prop.collision_drag*ammount_touching*velocity_diff.inner_product(offset);
            
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
        let dis_sqr = offset.dis_sqr();
        
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



fn convert_to_nan(r:Rect<f32>)->Rect<NotNaN<f32>>{

    let a=r.get_range(axgeom::XAXISS);
    let b=r.get_range(axgeom::YAXISS);
    
    Rect::new(
        NotNaN::new(a.left).unwrap(),
        NotNaN::new(a.right).unwrap(),
        NotNaN::new(b.left).unwrap(),
        NotNaN::new(b.right).unwrap()
        )
}
fn convert_from_nan(r:Rect<NotNaN<f32>>)->Rect<f32>{
    unsafe{std::mem::transmute(r)}
}


fn from_point(point:Vec2,radius:f32)->Rect<f32>{
    let r=radius;
    let point=point.0;
    Rect::new(point[0]-r,point[0]+r,point[1]-r,point[1]+r)
}

#[derive(Copy,Clone,Debug)]
pub struct Bot{
    pub pos: Vec2,
    pub vel: Vec2,
    pub acc: Vec2,
}
impl Bot{
    pub fn apply(&mut self,bot:&Bot){
        self.acc=bot.acc;
    }
    pub fn create_bbox(&self,bot_scene:&BotProp)->Rect<f32>{
        let p=self.pos.0;
        let r=bot_scene.radius.dis();
        Rect::new(p[0]-r,p[0]+r,p[1]-r,p[1]+r)
        //convert_to_nan(r)
    }
    pub fn create_bbox_nan(&self,bot_scene:&BotProp)->Rect<NotNaN<f32>>{
        convert_to_nan(self.create_bbox(bot_scene))
    }

    /*
    pub fn create_loose_bbox(&self,radius:f32)->Rect<NotNaN<f32>>{
        
        let mut r=convert_to_nan(from_point(self.pos,radius));


        let projected_pos=self.pos+self.vel;
        let r2=convert_to_nan(from_point(projected_pos,radius));

        r.grow_to_fit(&r2);
        r
    }
    */

    pub fn new(pos:Vec2)->Bot{
        let vel=Vec2([0.0;2]);
        let acc=vel;
        Bot{pos,vel,acc}
    }

    #[inline]
    pub fn pos(&self)->&Vec2{
        &self.pos
    }

    #[inline]
    pub fn vel(&self)->&Vec2{
        &self.vel
    }

    pub fn push_away(&mut self,b:&mut Self,radius:f32,max_amount:f32){
        let mut diff=b.pos-self.pos;

        let dis=diff.dis();

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
    pub midpoint:Vec2,
    pub rect:axgeom::Rect<f32>
}
impl Mouse{
    pub fn new(pos:Vec2,prop:&MouseProp)->Mouse{
        let mut m:Mouse=unsafe{std::mem::uninitialized()};
        m.mouse_prop= *prop;
        m.move_to(pos);
        m
    }

    pub fn get_rect(&self)->&axgeom::Rect<f32>{
        &self.rect
    }
    pub fn get_midpoint(&self)->&Vec2{
        &self.midpoint
    }
    pub fn get_radius(&self)->f32{
        self.mouse_prop.radius.dis()
    }
    pub fn move_to(&mut self,pos:Vec2){
        self.midpoint= pos;
        let p=self.midpoint.0;
        let r=self.mouse_prop.radius.dis();
        let r=axgeom::Rect::new(p[0]-r,p[0]+r,p[1]-r,p[1]+r);
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


