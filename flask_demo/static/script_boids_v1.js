////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Variables
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



let SPRITE_RADIUS = 10; // min 5 max 20
let PROTECTED_RANGE = 25; // min 0 max 500
let VISUAL_RANGE = 50;// min 0 max 500
let CURSOR_RANGE = 100;// min 0 max 500
let MARGIN = 150; // min 0 max 500

let MAX_SPEED = 2; // min 0.1 max 100
let MIN_SPEED = 0.1; // min 0.1 max 100

let AVOID_FACTOR = 0.01;// min 0 max 20
let TURN_FACTOR =  0.1;// min 0 max 20
let MATCHING_FACTOR = 0.01;// min 0 max 20
let CENTERING_FACTOR = 0.001;// min 0 max 20

let CURSOR_STRENGTH = -0.1; // min -1 max 1

let SET_FPS = 0; // zero means unlimited // min 0 max 100
let NUMBER_OF_BOIDS = 100;

let sprite_base_x = 0.7*SPRITE_RADIUS;
let sprite_base_y = Math.sqrt(SPRITE_RADIUS**2 - sprite_base_x**2);
const boid_array = [];

let PAUSE = false;
let CURSOR_DOWN = {
    pressed: false,
    x: 0,
    y: 0
  };

// Function to set input values from global variables
function setInputsFromGlobals() {
    document.getElementById("spriteRadiusInput").value = SPRITE_RADIUS;
    document.getElementById("protectedRangeInput").value = PROTECTED_RANGE;
    document.getElementById("visualRangeInput").value = VISUAL_RANGE;
    document.getElementById("marginInput").value = MARGIN;
    document.getElementById("maxSpeedInput").value = MAX_SPEED;
    document.getElementById("minSpeedInput").value = MIN_SPEED;
    document.getElementById("avoidFactorInput").value = AVOID_FACTOR;
    document.getElementById("turnFactorInput").value = TURN_FACTOR;
    document.getElementById("matchingFactorInput").value = MATCHING_FACTOR;
    document.getElementById("centeringFactorInput").value = CENTERING_FACTOR;
    document.getElementById("setFpsInput").value = SET_FPS;
    document.getElementById("numberOfBoids").value = NUMBER_OF_BOIDS
}

// Function to update global variables from input values
function updateGlobalsFromInputs() {
    SPRITE_RADIUS = parseFloat(document.getElementById("spriteRadiusInput").value);
    PROTECTED_RANGE = parseFloat(document.getElementById("protectedRangeInput").value);
    VISUAL_RANGE = parseFloat(document.getElementById("visualRangeInput").value);
    MARGIN = parseFloat(document.getElementById("marginInput").value);
    MAX_SPEED = parseFloat(document.getElementById("maxSpeedInput").value);
    MIN_SPEED = parseFloat(document.getElementById("minSpeedInput").value);
    AVOID_FACTOR = parseFloat(document.getElementById("avoidFactorInput").value);
    TURN_FACTOR = parseFloat(document.getElementById("turnFactorInput").value);
    MATCHING_FACTOR = parseFloat(document.getElementById("matchingFactorInput").value);
    CENTERING_FACTOR = parseFloat(document.getElementById("centeringFactorInput").value);
    SET_FPS = parseInt(document.getElementById("setFpsInput").value);
    NUMBER_OF_BOIDS = parseInt(document.getElementById("numberOfBoids").value)
    sprite_base_x = 0.7*SPRITE_RADIUS;
    sprite_base_y = Math.sqrt(SPRITE_RADIUS**2 - sprite_base_x**2);
}

// Attach event listeners to input elements to update globals when they change
document.querySelectorAll('.input-sync').forEach(input => {
    input.addEventListener('change', function () {
        updateGlobalsFromInputs();
    });
    input.addEventListener('input', function () {
        updateGlobalsFromInputs();
    });
});

// JavaScript code to reload the page when the button is clicked
document.getElementById("reloadButton").addEventListener("click", function() {
    location.reload(); // This reloads the current page
});

document.getElementById("pauseButton").addEventListener("click", function() 
{
    if (PAUSE)
    {
        PAUSE = false;
    }
    else
    {
        PAUSE = true;
    }
});

// Attach event listeners to add and remove boids
document.querySelectorAll('.n_boids').forEach(input => {
    input.addEventListener('input', function () 
    {
        if(boid_array.length > 0)
        {
            while (boid_array.length != NUMBER_OF_BOIDS)
            {
                if (boid_array.length < NUMBER_OF_BOIDS )
                {
                    const instance = new Boid();
                    boid_array.push(instance);
                }
                else
                {
                    boid_array.pop();
                }
            }
            
        }
    });
});


 
// Set the initial input values from global variables
setInputsFromGlobals();


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CANVAS SETUP
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const canvas = document.getElementById('canvas1');
let ctx = canvas.getContext('2d');
// Set canvas width and height to match its attributes
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

// Resize canvas
function resizeCanvas() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx = canvas.getContext('2d');
}
window.addEventListener('resize', resizeCanvas);


// mouse buttons

function logButtons(e) {
    if (e.buttons == 1)
    {
        CURSOR_DOWN.pressed = true;
        CURSOR_DOWN.x = e.x;
        CURSOR_DOWN.y = e.y;
    }
    else
    {
        CURSOR_DOWN.pressed = false;
    }
    // str = `${e.buttons} (${e.type}) (${e.x}) `;
    // console.log(str);
  }
  
document.addEventListener("mouseup", logButtons);
canvas.addEventListener("mousedown", logButtons);
canvas.addEventListener('mousemove', logButtons);



// create a boid

class Boid
{
    constructor()
    {
        // kinematics
        this.x = Math.random() * canvas.width/2 + canvas.width/4;
        this.y = Math.random() * canvas.height/2 + canvas.height/4;
        this.angle = Math.random()* Math.PI*2;
        this.vx = Math.random()*MAX_SPEED-MIN_SPEED+MIN_SPEED;
        this.vy = Math.random()*MAX_SPEED-MIN_SPEED+MIN_SPEED;

    }

    draw()
    {
        ctx.save();
        ctx.translate(this.x,this.y);
        ctx.rotate(this.angle);
        ctx.fillStyle = '#1d3f58';
        //ctx.drawImage(carImage, - SPRITE_RADIUS, - SPRITE_RADIUS*9/16 , SPRITE_RADIUS*2,SPRITE_RADIUS*2*9/16)
        ctx.beginPath();
        ctx.moveTo(SPRITE_RADIUS,0);
        ctx.lineTo(-sprite_base_x, sprite_base_y);
        ctx.lineTo(-sprite_base_x, -sprite_base_y);
        ctx.lineTo(SPRITE_RADIUS,0);
        ctx.fill();
        ctx.closePath();

        ctx.restore()
    }
}

// Generate Boids
for (let i = 0; i < NUMBER_OF_BOIDS; i++)
{
    const instance = new Boid();
    boid_array.push(instance);
}


var boid, otherboid, close_dx,close_dy, dist, speed,xvel_avg,yvel_avg,n, xpos_avg,ypos_avg ;

function handle_boids()
{
    for (let i = 0; i <boid_array.length; i++)
    {
        boid = boid_array[i];
        close_dx = 0;
        close_dy = 0;
        xvel_avg = 0;
        yvel_avg = 0;
        xpos_avg = 0;
        ypos_avg = 0; 

        n = 0;

        if (boid.x < MARGIN)
        {
            boid.vx = boid.vx + TURN_FACTOR
        }
        else if ((canvas.width - boid.x) < MARGIN)
        {
            boid.vx = boid.vx - TURN_FACTOR
        }

        if (boid.y < MARGIN)
        {
            boid.vy = boid.vy + TURN_FACTOR
        }
        else if ((canvas.height - boid.y) < MARGIN)
        {
            boid.vy = boid.vy - TURN_FACTOR
        }
        
        // cursor
        if (CURSOR_DOWN.pressed)
        {
            dist = Math.sqrt((boid.x-CURSOR_DOWN.x)**2 + (boid.y-CURSOR_DOWN.y)**2);
            if (dist < CURSOR_RANGE)
            {
                boid.vx += (CURSOR_DOWN.x - boid.x)*CURSOR_STRENGTH
                boid.vy += (CURSOR_DOWN.y - boid.y)*CURSOR_STRENGTH
            }

            ctx.beginPath();
            ctx.arc(CURSOR_DOWN.x, CURSOR_DOWN.y, CURSOR_RANGE, 0, Math.PI * 2); // Create a full circle
            ctx.stroke();
        }


        // loop through
        for (let j = 0; j < boid_array.length; j++)
        {
            if (j == i)
            {
                continue;
            }

            otherboid = boid_array[j];
            dist = Math.sqrt((boid.x-otherboid.x)**2 + (boid.y-otherboid.y)**2);
            
            if (dist < PROTECTED_RANGE)
            {
                close_dx += boid.x - otherboid.x
                close_dy += boid.y - otherboid.y
                //console.log(close_dx)
            }
            if (dist < VISUAL_RANGE)
            {
                xvel_avg += otherboid.vx
                yvel_avg += otherboid.vy
                xpos_avg += otherboid.x
                ypos_avg += otherboid.y
                n++;
            }


        }

        boid.vx += close_dx*AVOID_FACTOR
        boid.vy += close_dy*AVOID_FACTOR

        if (n>0)
        {
            boid.vx += (xvel_avg - boid.vx)*MATCHING_FACTOR
            boid.vy += (yvel_avg - boid.vy)*MATCHING_FACTOR

            xpos_avg /= n
            ypos_avg /= n

            boid.vx += (xpos_avg - boid.x)*CENTERING_FACTOR
            boid.vy += (ypos_avg - boid.y)*CENTERING_FACTOR

        }


        speed = Math.sqrt(boid.vx**2 + boid.vy**2);
        if (speed > MAX_SPEED)
        {
            boid.vx = (boid.vx/speed)*MAX_SPEED
            boid.vy = (boid.vy/speed)*MAX_SPEED
        }
        else if (speed < MIN_SPEED)
        {
            boid.vx = (boid.vx/speed)*MIN_SPEED
            boid.vy = (boid.vy/speed)*MIN_SPEED
        }

        
        boid.x += boid.vx
        boid.y += boid.vy
        boid.angle = Math.atan2(boid.vy,boid.vx)

        boid.draw();
    }
}


// Animation

// Frame Rate Calculation
let frameCount = 0;
let lastTime = performance.now();
let fps = 0;
function animate()
{
    // calculate frame Rate
    const currentTime = performance.now();
    frameCount++;
  
    if (currentTime - lastTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastTime = currentTime;
      document.getElementById("fpsValue").textContent = fps;
    }


    if (!PAUSE)
    {
        // Clear Canvas
        ctx.clearRect(0,0,canvas.width,canvas.height);

        // boids
        handle_boids();
    }

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