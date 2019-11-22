use axgeom::ordered_float::*;
use axgeom::*;
use dists;

///Properties that help express a group of bots.
///For example, their radius.
#[derive(Copy, Clone, Debug)]
pub struct BotProp {
    pub radius: Dist,
    pub collision_push: f32,
    pub collision_drag: f32,
    pub minimum_dis_sqr: f32,
    pub viscousity_coeff: f32,
}

impl BotProp {
    #[inline(always)]
    pub fn create_bbox_i32(&self, pos: Vec2<i32>) -> Rect<i32> {
        let p = pos;
        let r = self.radius.dis() as i32;
        Rect::new(p.x - r, p.x + r, p.y - r, p.y + r)
    }

    #[inline(always)]
    pub fn create_bbox(&self, pos: Vec2<f32>) -> Rect<f32> {
        let p = pos;
        let r = self.radius.dis();
        Rect::new(p.x - r, p.x + r, p.y - r, p.y + r)
    }

    #[inline(always)]
    pub fn create_bbox_nan(&self, pos: Vec2<f32>) -> Rect<NotNan<f32>> {
        self.create_bbox(pos).inner_try_into().unwrap()
    }

    #[inline(always)]
    pub fn liquid(&self, a: &mut Bot, b: &mut Bot) {
        let diff = b.pos - a.pos;

        let dis_sqr = diff.magnitude2();

        if dis_sqr < 0.0001 {
            a.acc += vec2(0.1, 0.0);
            b.acc -= vec2(0.1, 0.0);
            return;
        }

        if dis_sqr >= self.radius.dis2_squared() {
            return;
        }

        let dis = dis_sqr.sqrt();

        //d is zero if barely touching, 1 is overlapping.
        //d grows linearly with position of bots
        let d = 1.0 - (dis / (self.radius.dis2()));

        let spring_force_mag = -(d - 0.5) * 0.02;

        let velociy_diff = b.vel - a.vel;
        let damping_ratio = 0.0002;
        let spring_dampen = velociy_diff.dot(diff) * (1. / dis) * damping_ratio;

        let spring_force = diff * (1. / dis) * (spring_force_mag + spring_dampen);

        a.acc += spring_force;
        b.acc -= spring_force;
    }

    //#[inline(always)]
    pub fn collide(&self, bota: &mut Bot, botb: &mut Bot) {
        //Takes a values between 0 and 1, and returns a value between 0 and 1.
        //The input is the distance from not touching.
        //So if the bots are just barely touching, the input will be 0.
        //If they are right on top of each other, the input will be 1.

        //The output is the normalized force with which to handle.
        //A value of 0 is no force.
        //A value of 1 is full force.
        #[inline(always)]
        pub fn handle_repel(input: f32) -> f32 {
            let a = 3.0 * input * input;
            a.min(1.0)
        }

        let prop = self;

        let offset = bota.pos - botb.pos;

        let dis_sqr = offset.magnitude2();

        if dis_sqr >= prop.radius.dis2_squared() {
            //They not touching (bots are circular).
            return;
        }

        if dis_sqr < 0.00001 {
            bota.acc += vec2(0.1, 0.1);
            botb.acc -= vec2(0.1, 0.1);
            return;
        }

        //At this point, we know they collide!!!!

        /*
        fn fast_inv_sqrt(x: f32) -> f32 {
            let i: u32 = unsafe { std::mem::transmute(x) };
            let j = 0x5f3759df - (i >> 1);
            let y: f32 = unsafe { std::mem::transmute(j) };
            y * (1.5 - 0.5 * x * y * y)
        }
        */

        let sum_rad = prop.radius.dis2();

        let disinv = 1.0 / dis_sqr.sqrt();
        //let disinv=fast_inv_sqrt(dis_sqr);

        //0 if barely touching (not touching)
        //1 if directly on top of each other
        //let dd=(sum_rad-dis)/sum_rad;
        //let dd=sum_rad/sum_rad - dis/sum_rad;
        //let dd=1 - dis/sum_rad;
        let dd = 1.0 - 1.0 / (sum_rad * disinv);

        let ammount_touching = handle_repel(dd);

        let push_mag = ammount_touching * prop.collision_push;


        let velocity_diff = bota.vel - botb.vel;

        let drag = -prop.collision_drag * ammount_touching * velocity_diff.dot(offset);

        let push1 = drag + push_mag;
        let push2 = -drag - push_mag;

        let push_force1 = offset * (push1 * disinv);
        let push_force2 = offset * (push2 * disinv);

        let viscous = velocity_diff * (-prop.viscousity_coeff * ammount_touching);

        bota.acc += push_force1;
        //bota.acc += viscous;

        botb.acc += push_force2;
        //botb.acc += viscous;
    }

    #[inline(always)]
    pub fn collide_mouse(&self, bot: &mut Bot, mouse: &Mouse) {
        let prop = self;
        let offset = *mouse.get_midpoint() - bot.pos;
        let dis_sqr = offset.magnitude2();

        let sum_rad = mouse.get_radius() + prop.radius.dis();
        if dis_sqr < sum_rad * sum_rad {
            let dis = dis_sqr.sqrt();

            if dis < 0.0001 {
                return;
            }

            let vv = (sum_rad - dis) / sum_rad;
            let vv = vv * vv;
            let vv2 = (5.0 * vv).min(1.0);

            let push_mag = vv2 * mouse.mouse_prop.force;
            let push_force = offset * (push_mag / dis);

            bot.acc += -push_force;
        }
    }
}

///A bot with 2d position,velocity,acceleration
#[derive(Copy, Clone, Debug)]
pub struct Bot {
    pub pos: Vec2<f32>,
    pub vel: Vec2<f32>,
    pub acc: Vec2<f32>,
}

impl crate::BorderCollideTrait for Bot {
    type N = f32;
    #[inline(always)]
    fn pos_vel_mut(&mut self) -> (&mut Vec2<f32>, &mut Vec2<f32>) {
        (&mut self.pos, &mut self.vel)
    }
}
impl Bot {

    #[inline(always)]
    pub fn move_to_point(&mut self,target:Vec2<f32>,radius:f32) -> bool{
        let diff=target-self.pos-self.vel*40.0;
        let lens=diff.magnitude2();
        if lens>0.001{
            self.acc+=diff*(0.03/lens.sqrt());
        }
        lens<radius
    }

    #[inline(always)]
    pub fn update(&mut self){
        let bot=self;
        bot.pos+=bot.vel;
        bot.vel+=bot.acc;    
        bot.acc=vec2(0.0,0.0);
    }
    #[inline(always)]
    pub fn create_bbox(&self, bot_scene: &BotProp) -> Rect<f32> {
        let p = self.pos;
        let r = bot_scene.radius.dis();
        Rect::new(p.x - r, p.x + r, p.y - r, p.y + r)
    }

    #[inline(always)]
    pub fn create_bbox_nan(&self, bot_scene: &BotProp) -> Rect<NotNan<f32>> {
        self.create_bbox(bot_scene).inner_try_into().unwrap()
    }

    #[inline(always)]
    pub fn new(pos: Vec2<f32>) -> Bot {
        let vel = vec2(0.0, 0.0);
        let acc = vec2(0.0, 0.0);
        Bot { pos, vel, acc }
    }

    #[inline(always)]
    pub fn push_away(&mut self, b: &mut Self, radius: f32, max_amount: f32) {
        let mut diff = b.pos - self.pos;

        let dis = diff.magnitude2().sqrt();

        if dis < 0.000_001 {
            return;
        }

        let mag = max_amount.min(radius * 2.0 - dis);
        if mag < 0.0 {
            return;
        }
        //let mag=max_amount;
        diff *= mag / dis;

        self.acc -= diff;
        b.acc += diff;

        //TODO if we have moved too far away, move back to point of collision!!!
        {}
    }
}

///Properties of a mouse
#[derive(Copy, Clone, Debug)]
pub struct MouseProp {
    pub radius: Dist,
    pub force: f32,
}

///Mouse object. Includes position of the mouse and its aabb.
#[derive(Copy, Clone, Debug)]
pub struct Mouse {
    pub mouse_prop: MouseProp,
    pub midpoint: Vec2<f32>,
    pub rect: axgeom::Rect<f32>,
}
impl Mouse {
    #[inline(always)]
    pub fn new(midpoint: Vec2<f32>, mouse_prop: MouseProp) -> Mouse {
        let r = vec2same(mouse_prop.radius.dis());
        Mouse {
            mouse_prop,
            midpoint,
            rect: Rect::from_point(midpoint, r),
        }
    }

    #[inline(always)]
    pub fn get_rect(&self) -> &axgeom::Rect<f32> {
        &self.rect
    }

    #[inline(always)]
    pub fn get_midpoint(&self) -> &Vec2<f32> {
        &self.midpoint
    }

    #[inline(always)]
    pub fn get_radius(&self) -> f32 {
        self.mouse_prop.radius.dis()
    }

    #[inline(always)]
    pub fn move_to(&mut self, pos: Vec2<f32>) {
        self.midpoint = pos;
        let p = self.midpoint;
        let r = self.mouse_prop.radius.dis();
        let r = axgeom::Rect::new(p.x - r, p.x + r, p.y - r, p.y + r);
        self.rect = r;
    }
}

///A struct with cached calculations based off of a radius.
#[derive(Copy, Clone, Debug)]
pub struct Dist {
    dis: f32,
    dis2: f32,
    dis2_squared: f32,
}
impl Dist {
    #[inline(always)]
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
    #[inline(always)]
    pub fn dis(&self) -> f32 {
        self.dis
    }

    ///Returns the cached radius*2.0
    #[inline(always)]
    pub fn dis2(&self) -> f32 {
        self.dis2
    }

    ///Returns the cached radius.powi(2)
    #[inline(always)]
    pub fn dis2_squared(&self) -> f32 {
        self.dis2_squared
    }
}

///A group of bots and a property object describing them.
pub struct BotScene<T> {
    pub bot_prop: BotProp,
    pub bots: Vec<T>,
}

///Builder to build a bot scene.
#[derive(Copy, Clone, Debug)]
pub struct BotSceneBuilder {
    grow: f32,
    radius: f32,
    num: usize,
    bot_radius: f32,
}
impl BotSceneBuilder {
    pub fn new(num: usize) -> BotSceneBuilder {
        BotSceneBuilder {
            grow: 0.2,
            radius: 17.0,
            num,
            bot_radius: 5.0,
        }
    }

    pub fn with_grow(&mut self, grow: f32) -> &mut Self {
        self.grow = grow;
        self
    }
    pub fn with_num(&mut self, num: usize) -> &mut Self {
        self.num = num;
        self
    }

    pub fn with_radius_of(&mut self, radius: f32) -> &mut Self {
        self.radius = radius;
        self
    }

    pub fn build_specialized<T>(&mut self, mut func: impl FnMut(Vec2<f32>) -> T) -> BotScene<T> {
        let spiral = dists::spiral::Spiral::new([0.0, 0.0], self.radius, self.grow);

        let bots: Vec<T> = spiral.take(self.num).map(|pos| func(pos)).collect();

        let bot_prop = BotProp {
            radius: Dist::new(self.bot_radius),
            collision_push: 0.1,
            collision_drag: 0.1,
            minimum_dis_sqr: 0.0001,
            viscousity_coeff: 0.1,
        };

        BotScene { bot_prop, bots }
    }
    pub fn build(&mut self) -> BotScene<Bot> {
        let spiral = dists::spiral::Spiral::new([0.0, 0.0], self.radius, self.grow);

        let bots: Vec<Bot> = spiral.take(self.num).map(|pos| Bot::new(pos)).collect();

        let bot_prop = BotProp {
            radius: Dist::new(self.bot_radius),
            collision_push: 0.1,
            collision_drag: 0.1,
            minimum_dis_sqr: 0.0001,
            viscousity_coeff: 0.1,
        };

        BotScene { bot_prop, bots }
    }
}
