////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CANVAS SETUP
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


const canvas = document.getElementById('canvas1');
let ctx = canvas.getContext('2d');
// Set canvas width and height to match its attributes
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;
let COORDINATE_CONVERSION_SCALE = Math.min((canvas.height-5) / 4 , (canvas.width-5) / 4);
let CANVAS_BOUNDS = canvas.getBoundingClientRect();
//console.log("COORDINATE_CONVERSION_SCALE: " + COORDINATE_CONVERSION_SCALE);

// Resize canvas
function resizeCanvas() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx = canvas.getContext('2d');
    COORDINATE_CONVERSION_SCALE = Math.min((canvas.height-5) / 4 , (canvas.width-5) / 4);
    CANVAS_BOUNDS = canvas.getBoundingClientRect();
}
window.addEventListener('resize', resizeCanvas);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let FPS = 0;
let SET_FPS = 10;
let CURSOR_DOWN = {
    pressed: false,
    x: 0,
    y: 0
};


let ENVIROMENT_RADIUS = 2;
let ENVIROMENT_COLOR = '#1d3f58';
let ENVIROMENT_THICKNESS = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EVENT LISTENERS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function logButtons(e) {
    if (e.buttons == 1)
    {
        CURSOR_DOWN.pressed = true;
        CURSOR_DOWN.x = e.x - CANVAS_BOUNDS.left;
        CURSOR_DOWN.y = e.y - CANVAS_BOUNDS.top;
    }
    else
    {
        CURSOR_DOWN.pressed = false;
    }
    // str = `${e.buttons} (${e.type})  x: (${e.x}) , y: (${e.y})`;
    // console.log(str);
  }
  
document.addEventListener("mouseup", logButtons);
canvas.addEventListener("mousedown", logButtons);
canvas.addEventListener('mousemove', logButtons);

function canvas_to_cartesian(x, y)
{
    return [(x - CANVAS_BOUNDS.width / 2) / COORDINATE_CONVERSION_SCALE, -(y - CANVAS_BOUNDS.height / 2) / COORDINATE_CONVERSION_SCALE];
}

function cartesian_to_canvas(x, y)
{
    return [x * COORDINATE_CONVERSION_SCALE + CANVAS_BOUNDS.width / 2, -y * COORDINATE_CONVERSION_SCALE + CANVAS_BOUNDS.height / 2];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DRAWING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function drawEnvironment(enviroment_radius, enviroment_color, enviroment_thickness)
{
    // Calculating the coordinates of the pentagon vertices
    let x1 = Math.sin(0 * (2 * Math.PI / 5)) * enviroment_radius;
    let y1 = Math.cos(0 * (2 * Math.PI / 5)) * enviroment_radius;
    
    let x2 = Math.sin(1 * (2 * Math.PI / 5)) * enviroment_radius;
    let y2 = Math.cos(1 * (2 * Math.PI / 5)) * enviroment_radius;
    
    let x3 = Math.sin(2 * (2 * Math.PI / 5)) * enviroment_radius;
    let y3 = Math.cos(2 * (2 * Math.PI / 5)) * enviroment_radius;
    
    let x4 = Math.sin(3 * (2 * Math.PI / 5)) * enviroment_radius;
    let y4 = Math.cos(3 * (2 * Math.PI / 5)) * enviroment_radius;
    
    let x5 = Math.sin(4 * (2 * Math.PI / 5)) * enviroment_radius;
    let y5 = Math.cos(4 * (2 * Math.PI / 5)) * enviroment_radius;

    // console.log("x1: " + x1 + " y1: " + y1);
    
    [x1, y1] = cartesian_to_canvas(x1, y1);
    [x2, y2] = cartesian_to_canvas(x2, y2);
    [x3, y3] = cartesian_to_canvas(x3, y3);
    [x4, y4] = cartesian_to_canvas(x4, y4);
    [x5, y5] = cartesian_to_canvas(x5, y5);

    // console.log("x1: " + x1 + " y1: " + y1);
    // console.log("width: " + CANVAS_BOUNDS.width + " height: " + CANVAS_BOUNDS.height);

    // Drawing the pentagon
    ctx.lineWidth = enviroment_thickness;
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.lineTo(x3,y3);
    ctx.lineTo(x4,y4);
    ctx.lineTo(x5,y5);
    //ctx.lineTo(x1,y1);
    ctx.closePath();
    ctx.strokeStyle = enviroment_color;
    ctx.stroke();

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AGENTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Agent
{
    constructor(x, y,vx,vy, radius, color, type )
    {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.radius = radius;
        this.color = color;
        this.type = type;
    }

    draw()
    {
        let [x, y] = cartesian_to_canvas(this.x, this.y);
        ctx.beginPath();
        ctx.arc(x, y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ANIMATION
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Frame Rate Calculation
let frameCount = 0;
let lastTime = performance.now();

// create the agent
let agent = new Agent(0, 0, 0, 0, 5, '#163145', 'p');

function animate()
{
    // calculate frame Rate
    const currentTime = performance.now();
    frameCount++;
  
    if (currentTime - lastTime >= 1000) {
      FPS = frameCount;
      frameCount = 0;
      lastTime = currentTime;
      //console.log(FPS);
    }

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the enviroment
    drawEnvironment(ENVIROMENT_RADIUS, ENVIROMENT_COLOR, ENVIROMENT_THICKNESS);

    // Draw the agents
    agent.draw();


    if (SET_FPS > 0)
    {
        setTimeout(() => {
            requestAnimationFrame(animate);
          }, 1000 / SET_FPS);
    }
    else
    {
        requestAnimationFrame(animate);    
    }
}

requestAnimationFrame( animate );