use crate::axgeom::*;
use bit_vec::*;

#[derive(Copy,Clone,Debug,Eq,PartialEq)]
pub enum CardDir{
    U,
    D,
    L,
    R
}
impl CardDir{
    pub fn into_char(self)->char{
        use CardDir::*;
        match self{
            U=>{
                '↑'
            },
            D=>{
                '↓'
            },
            L=>{
                '←'
            },
            R=>{
                '→'
            }
        }
    }
    pub fn into_vec(self)->Vec2<GridNum>{
        use CardDir::*;
        match self{
            U=>{
                vec2(0,-1)
            },
            D=>{
                vec2(0,1)
            },
            L=>{
                vec2(-1,0)
            },
            R=>{
                vec2(1,0)
            }
        }
    }
    pub fn into_two_bits(self)->u8{
        use CardDir::*;
        match self{
            U=>{
                0b00
            },
            D=>{
                0b01
            },
            L=>{
                0b10
            },
            R=>{
                0b11
            }

        }
    }
    
}


#[derive(Copy,Clone)]
pub struct Iterator2D{
    counter:Vec2<GridNum>,
    dim:Vec2<GridNum>
}
impl Iterator2D{
    pub fn new(dim:Vec2<GridNum>)->Iterator2D{
        Iterator2D{counter:vec2same(0),dim}
    }
}

use core::iter::*;

impl FusedIterator for Iterator2D{}
impl ExactSizeIterator for Iterator2D{}
impl Iterator for Iterator2D{
    type Item=Vec2<GridNum>;
    fn size_hint(&self)->(usize,Option<usize>){
        let diff = vec2(self.dim.x,self.dim.y-1)-self.counter;
        //TODO test this
        let l=(self.dim.x*diff.y+diff.x) as usize;
        (l,Some(l))
    }
    fn next(&mut self)->Option<Self::Item>{

        if self.counter.y==self.dim.y{
            return None
        }

        let k=self.counter;

        self.counter.x+=1;
        if self.counter.x==self.dim.x{
            self.counter.x=0;
            self.counter.y+=1;
        }
        Some(k)
    }
}

#[test]
fn test_iterator2d(){
    let i=Iterator2D::new(vec2(10,20));
    assert_eq!(i.len(),200);
    assert_eq!(i.count(),200);



    let i=Iterator2D::new(vec2(20,10));
    assert_eq!(i.len(),200);
    assert_eq!(i.count(),200);

}


impl<'a> FusedIterator for CellIterator<'a>{}
impl<'a> ExactSizeIterator for CellIterator<'a>{}
pub struct CellIterator<'a>{
    grid:&'a Grid2D,
    inner:Iterator2D
}
impl<'a> Iterator for CellIterator<'a>{
    type Item=(Vec2<GridNum>,bool);

    fn size_hint(&self)->(usize,Option<usize>){
        self.inner.size_hint()
    }
    fn next(&mut self)->Option<Self::Item>{
        match self.inner.next(){
            Some(v)=>{
                Some((v,self.grid.get(v)))
            },
            None=>{
                None
            }
        }
    }

}



pub type GridNum=isize;

pub struct Grid2D {
    dim: Vec2<GridNum>,
    inner: BitVec,
}

impl Grid2D {
    pub fn from_str(dim:Vec2<GridNum>,p:&str)->Grid2D{
        let mut grid=Grid2D::new(dim);

        for (y,line) in p.lines().enumerate(){
            for (x,c) in line.chars().enumerate(){
                match c{
                    '█'=>{
                        grid.set(vec2(x,y).inner_as(),true);
                    },
                    ' '=>{

                    }
                    _=>{
                        panic!("unknown char");
                    }
                }
            }
        }
        grid

    }
    pub fn new(dim:Vec2<GridNum>) -> Grid2D {
        let inner = BitVec::from_elem((dim.x * dim.y) as usize, false);

        Grid2D { dim, inner }
    }
    pub fn dim(&self)->Vec2<GridNum>{
        self.dim
    }

    pub fn iter(&self)->CellIterator{

        let inner = Iterator2D::new(self.dim);
        CellIterator{grid:self,inner}
    }
    pub fn get(&self, p:Vec2<GridNum>) -> bool {
        self.inner[(p.x * self.dim.y + p.y) as usize]
    }
    pub fn get_option(&self, p:Vec2<GridNum>) -> Option<bool> {
        self.inner.get((p.x * self.dim.y + p.y) as usize)
    }
    pub fn set(&mut self, p:Vec2<GridNum>,val:bool) {
        self.inner.set( (p.x * self.dim.y + p.y) as usize, val)
    }
    pub fn len(&self)->usize{
        (self.dim.x*self.dim.y) as usize
    }

    pub fn draw_map(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let mut res=String::new();
        
        let mut fa=Ok(());



        fa.and(writeln!(f,""));
        for i in 0..self.dim().y{
            for j in 0..self.dim().x{
                
                let cc=if self.get(vec2(j,i)){
                    "1 "
                }else{
                    "0 "
                };

                res.push_str(cc);
            }
            fa=fa.and( writeln!(f,"{}",res));
            res.clear();
        }
        fa
    }


}


pub type WorldNum=f32;


#[test]
fn testy(){
    let mut inner=Grid2D::new(vec2(20,20));
    let k =GridDim2D{dim:Rect::new(-100.0,100.0,-100.0,100.0),inner};

    let j = k.convert_to_grid(vec2(56.0,56.0));
    
    dbg!(j);
    
    let back=k.convert_to_world(j);
    assert_eq!(back,vec2(50.0,50.0));
    //dbg!(back);
    //panic!("yo");
}


use core::fmt;
impl fmt::Debug for Grid2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.draw_map(f)
    }
}


pub fn pick_empty_spot(grid:&Grid2D)->Option<Vec2<GridNum>>{
    //TODO inefficient
    use rand::prelude::*;

    let mut k:Vec<_>=Iterator2D::new(grid.dim()).filter(|a|!grid.get(*a)).collect();

    let mut rng = rand::thread_rng();
    k.shuffle(&mut rng);

    k.first().map(|a|*a)
}

pub fn find_closest_empty(grid:&Grid2D,start:Vec2<GridNum>)->Option<Vec2<GridNum>>{
    //TODO inefficient.

    let mut k:Vec<_>=Iterator2D::new(grid.dim()).filter(|a|!grid.get(*a)).map(|a|(a,(start-a).magnitude2() )).collect();

    k.sort_by(|a,b|a.1.cmp(&b.1));

    k.first().map(|a|a.0)
}

#[derive(Copy,Clone,Debug)]
pub enum GridRayCastResult{
    Found{t:WorldNum,cell:Vec2<GridNum>,dirhit:CardDir},
    NotFound
}


pub mod raycast{
    use core::iter::*;
    use crate::grid::*;
    use crate::Ray;

    #[derive(Copy,Clone,Debug)]
    pub struct CollideCellEvent{
        //Cell colliding with
        pub cell:Vec2<GridNum>,
        //Direction in which we are colliding with it.
        pub dir_hit:CardDir,

        //So the user can see how long the ray is now.
        pub tval:WorldNum
    }
    pub struct RayCaster<'a>{
        grid:&'a GridViewPort,
        ray:Ray<WorldNum>,
        dir_sign:Vec2<GridNum>,
        next_dir_sign:Vec2<GridNum>,
        current_grid:Vec2<GridNum>,
        tval:WorldNum
    }
    impl<'a> RayCaster<'a>{
        pub fn new(grid:&'a GridViewPort,ray:Ray<WorldNum>)->Option<RayCaster>{
            let dir_sign=vec2(if ray.dir.x>0.0{1}else{0},if ray.dir.y>0.0{1}else{0});
            let next_dir_sign=vec2(if ray.dir.x>0.0{1}else{-1},if ray.dir.y>0.0{1}else{-1});
            
            let current_grid=grid.to_grid(ray.point);


            if ray.dir.magnitude2()>0.0{
                Some(RayCaster{grid,ray,dir_sign,next_dir_sign,current_grid,tval:0.0})
            }else{
                None
            }
        }
    }
    impl FusedIterator for RayCaster<'_>{}
    impl Iterator for RayCaster<'_>{
        type Item=CollideCellEvent;
        fn next(&mut self)->Option<Self::Item>{
            let grid=&self.grid;
            let ray=&self.ray;
            let dir_sign=self.dir_sign;

            let next_grid=self.current_grid+dir_sign;
            let next_grid_pos=grid.to_world_topleft(next_grid);


            //A ray can be described as follows:
            //rx(t)=ray.dir.x*tval+ray.point.x
            //ry(t)=ray.dir.y*tval+ray.point.y
            //
            //The ray itself is all the points that satify those two equations,
            //where tval>0.
            //
            //As tval increases, so does the ray length.
            //
            //
            //We want to find out when a ray intersects
            //th next cell. A ray are intersect the cell either on
            //a x axis or in a y axis.
            //so we have two equations.
            //
            //Equation for when it hits the xaxis  
            //next_grid_pos.x=ray.dir.x*tval+ray.point.x
            //
            //Equation for when it hits the yaxis
            //next_grid_pos.y=ray.dir.y*tval+ray.point.y
            //
            //In both cases, lets solve for tval.
            //The equation with the smaller tval is the one
            //the ray will hit first.
            //
            //If the tval for the x equation is smaller, the ray
            //will intersect the Y axis first.
            //

            let tvalx=(next_grid_pos.x-ray.point.x)/ray.dir.x;
            let tvaly=(next_grid_pos.y-ray.point.y)/ray.dir.y;

            
            //dbg!(tvalx,tvaly);
            //TODO test that this all works with negative numbers!!!
            /*
            if tvalx.is_finite(){
                assert_ge!(tvalx,0.0,"{:?}",(ray,self.current_grid,next_grid,next_grid_pos));
            }
            if tvaly.is_finite(){
                assert_ge!(tvaly,0.0,"{:?}",(ray,self.current_grid,next_grid,next_grid_pos));
            }
            */

            let mut dir_hit;
            if (tvalx.is_finite() && tvalx<=tvaly) || tvaly.is_infinite() || tvaly.is_nan(){
                if dir_sign.x==1{
                    //hit left side
                    dir_hit=CardDir::L;
                    //1
                }else{
                    dir_hit=CardDir::R;
                    //hit right side
                }
                self.tval=tvalx;
                self.current_grid.x+=self.next_dir_sign.x;
            }else if tvaly<tvalx  || tvalx.is_infinite() || tvalx.is_nan(){

                if dir_sign.y==1{
                    //hit top side
                    dir_hit=CardDir::U;
                }else{
                    //hit bottom side
                    dir_hit=CardDir::D;
                }
                self.tval=tvaly;
                self.current_grid.y+=self.next_dir_sign.y;
            }else{
                unreachable!("{:?}, {:?}",(tvalx,tvaly),ray);
            }
            Some(CollideCellEvent{tval:self.tval,cell:self.current_grid,dir_hit})
        }
    }
}





pub struct GridViewPort{
    pub spacing:Vec2<WorldNum>,
    pub origin:Vec2<WorldNum>
}
impl GridViewPort{

    pub fn to_world_topleft(&self,pos:Vec2<GridNum>)->Vec2<WorldNum>{
        pos.inner_as().scale(self.spacing)+self.origin
    }

    pub fn to_world_center(&self,pos:Vec2<GridNum>)->Vec2<WorldNum>{
        pos.inner_as().scale(self.spacing)+self.origin+self.spacing/2.0
    }
    
    pub fn to_grid(&self,pos:Vec2<WorldNum>)->Vec2<GridNum>{
    
        let result = (pos-self.origin).inv_scale(self.spacing);

        result.inner_as()


        /*
        let xdim = self.grid_dim.x;
        let ydim = self.grid_dim.y;

        let dim: &Rect<f32> = &self.dim;

        let pos=pos-dim.top_left();
        //https://math.stackexchange.com/questions/528501/how-to-determine-which-cell-in-a-grid-a-point-belongs-to
        
        let x = pos.x;
        let y = pos.y;
        let width = dim.x.right - dim.x.left;
        let height = dim.y.right - dim.y.left;

        let i = (x * (xdim as f32 / width))
            .floor()
            .max(0.0)
            .min((xdim - 1) as f32);
        let j = (y * (ydim as f32 / height))
            .floor()
            .max(0.0)
            .min((ydim - 1) as f32);
        let i = i as isize;
        let j = j as isize;

       
        vec2(i,j)
        */
    }


    pub fn cell_radius(&self)->Vec2<WorldNum>{
        self.spacing
        /*
        let spacingx=(self.dim.x.right-self.dim.x.left)/self.grid_dim.x as f32;
        let spacingy=(self.dim.y.right-self.dim.y.left)/self.grid_dim.y as f32;
        vec2(spacingx,spacingy)
        */
    }
    /*
    pub fn convert_to_world_topleft(&self,val:Vec2<GridNum>)->Result<Vec2<WorldNum>,ToWorldError>{
        if val.x>=self.grid_dim.x || val.y>=self.grid_dim.y{
            Err(ToWorldError{dim:self.grid_dim,pos:val})
        }else{
            let top_left=vec2(self.dim.x.left,self.dim.y.left);

            let spacingx=(self.dim.x.right-self.dim.x.left)/self.grid_dim.x as f32;
            let spacingy=(self.dim.y.right-self.dim.y.left)/self.grid_dim.y as f32;
            

            let val=vec2(spacingx * val.x as f32,spacingy*val.y as f32);
            //let half=vec2(spacingx,spacingy)/2.0;
            Ok(top_left+val)
        }
    }
    pub fn convert_to_world_center(&self,val:Vec2<GridNum>)->Result<Vec2<WorldNum>,ToWorldError>{
        if val.x>=self.grid_dim.x || val.y>=self.grid_dim.y{
            Err(ToWorldError{dim:self.grid_dim,pos:val})
        }else{
            let top_left=vec2(self.dim.x.left,self.dim.y.left);

            let spacingx=(self.dim.x.right-self.dim.x.left)/self.grid_dim.x as f32;
            let spacingy=(self.dim.y.right-self.dim.y.left)/self.grid_dim.y as f32;
            

            let val=vec2(spacingx * val.x as f32,spacingy*val.y as f32);
            let half=vec2(spacingx,spacingy)/2.0;
            Ok(top_left+val+half)
        }
    }
    */
}



