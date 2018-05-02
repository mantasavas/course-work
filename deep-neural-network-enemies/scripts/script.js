/*Tanka data*/
tank_two_x = 60;
tank_two_y = 400;
tank_two_speed = 5;
tank_two_angle = 0;
tank_two_mode = 0;


/*Rectangle data*/
tank_one_x = 0;
tank_one_y = 0;
tank_one_speed = 5;
tank_one_angle = 0;
tank_one_mode = 0;




game_window_size_x = 1300;
game_window_size_y = 800;


class Vector
{
    constructor(x = 0, y = 0)
    {
        this.x = x;
        this.y = y;
    }
}


class Circle
{
    constructor(x_cordinate, y_cordinate, turn_angle, color)
    {
        /****** Circle center cordinates ******/
        this.x_cordinate = x_cordinate;
        this.y_cordinate = y_cordinate;

        // Which angle circle is turned to
        this.turn_angle = turn_angle;

        // Set color of circle
        this.color = color;

        /****** Cordinates for collision ******/
        // Right Bottom
        this.triangle_rax = 0;
        this.triangle_ray = 0;
        // Left Botttom
        this.triangle_lbx = 0;
        this.trinagle_lby = 0;
        // Top
        this.triangle_tkx = 0;
        this.triangle_tky = 0;


        // If it's 1 move forward, -1 backwards
        this.moving_mode = 0;
        // How fast object moves forward, direction depens on mode above
        this.moving_speed = 5;
        // Current vision length
        this.current_vision = 300;
        // Max vision length
        this.max_sight_distance = 400;
        // Min vision length
        this.min_sight_distance = 70;


        /*Shooting*/
        // When shooting can be done again
        this.shooting_time = 100;
        this.last_degree_before_rotation = 0;
        this.pos_before_shotx = 0;
        this.pos_before_shoty = 0;
        this.shooted = false;
        this.projectile_x = 0;
        this.projectile_y = 22;
        this.projectile_eqb = 0;
        this.projectile_eqm = 0;
        this.number_of_hits = 0;
        this.proj_transf_x = 0;
        this.proj_transf_y = 0;
    }
}


var circles = [];
circles.push(new Circle(150, 600, 0, "#995f3b"));
circles.push(new Circle(1150, 200, 180, "#99dcf7"));


/*Circle*/
circle_one_x = 400;
circle_one_y = 400;
circle_one_speed = 5;
circle_one_angle = 0;
circle_one_mode = 0;
circle_one_long_vision = 300;
circle_max_sight_distance = 400;
circle_min_sight_distance = 70;



/*tank keys*/
var circle_two_keys = {
    a_key: false,
    s_key: false,
    d_key: false,
    w_key: false,
    o_key: false,
    p_key: false,
    spacebar_key: false
};

/*rectangular keys*/
var circle_one_keys = {
    up_key: false,
    down_key: false,
    left_key: false,
    right_key: false,
    z_key: false,
    x_key: false,
    one_key: false
};

canvas = document.getElementById("canvas");

//console.log(canvas);

context = canvas.getContext("2d");


tank_one_image = new Image();
tank_one_image.src = "https://imgur.com/C7UqRZs.jpg";


tank_two_image = new Image();
tank_two_image.src = "https://imgur.com/C7UqRZs.jpg";

window.addEventListener("keyup", keyup_handler, false);

var moveInterval = setInterval(function () {
    draw();
}, 30);

function draw() {
    context = canvas.getContext("2d");
    context.clearRect(0, 0, game_window_size_x, game_window_size_y);

    context.fillStyle = "rgb(200, 100, 220)";

    setWeights();

    drawEntity(0);
    drawEntity(1);


}


/*Checks if border has been crossed, if true => stop game and start again*/
function crossedBorder(x_cordinate, y_cordinate, circle_index)
{
    //console.log(x_cordinate + " : " + y_cordinate);

    if(x_cordinate < 0 || y_cordinate < 0 || x_cordinate > game_window_size_x || y_cordinate > game_window_size_y)
    {
        return true;
    }
    else if( (circle_index == 0) && (x_cordinate > game_window_size_x / 2))
    {
        return true;
    }
    else if((circle_index == 1) && (x_cordinate < game_window_size_x / 2))
    {
        return true;
    }
    else
    {
        return false;
    }

}

function resetGame()
{
    /*Reseting cordinates and degrees of the first enemy*/
    circles[0].x_cordinate = 150;
    circles[0].y_cordinate = 600;
    circles[0].turn_angle = 0;

    /*Reseting cordinates and degrees of the second enemy*/
    circles[1].x_cordinate = 1150;
    circles[1].y_cordinate = 200;
    circles[1].turn_angle = 180;
}

function drawEntity(circle_index)
{

    /*Finds enemy index*/
    if(circle_index == 0)
        deter_index = 1;
    else
        deter_index = 0;


    if(crossedBorder(circles[circle_index].x_cordinate, circles[circle_index].y_cordinate, circle_index))
    {
        circles[circle_index].number_of_hits -= 5;
        resetGame();
    }

    /*Sets fixed vision interval, stop from exeeding*/
    if(circles[circle_index].current_vision < circles[circle_index].min_sight_distance)
    {
        circles[circle_index].current_vision = circles[circle_index].min_sight_distance;
    }
    else if(circles[circle_index].current_vision > circles[circle_index].max_sight_distance)
    {
        circles[circle_index].current_vision = circles[circle_index].max_sight_distance;
    }

    // Formula to calculate vision wide, as vision get longer widness decresses
    var sight_wide = ( (circles[circle_index].max_sight_distance + 100) - circles[circle_index].current_vision) * 0.3;
    var degrees = Math.atan(sight_wide / circles[circle_index].current_vision) * (180 / Math.PI) * 2;

    /*Circles cordinates*/
    small_circle_x = 0;
    small_circle_y = 22;

    big_circle_x = 0;
    big_circle_y = 0;

    /*Triangle cordinates*/
    tright_corner_x = sight_wide;
    tright_corner_y = circles[circle_index].current_vision;

    ttop_corner_x = 0;
    ttop_corner_y = 31;

    tleft_corner_x = -sight_wide;
    tleft_corner_y = circles[circle_index].current_vision;


    /*Only works if user wants to move object, calculates moving cordinates*/
    circles[circle_index].x_cordinate += (circles[circle_index].moving_speed * circles[circle_index].moving_mode) * Math.cos(Math.PI / 180 * circles[circle_index].turn_angle);
    circles[circle_index].y_cordinate += (circles[circle_index].moving_speed * circles[circle_index].moving_mode) * Math.sin(Math.PI / 180 * circles[circle_index].turn_angle);



    /*Changes axis*/
    context.save();
    context.translate(circles[circle_index].x_cordinate, circles[circle_index].y_cordinate);
    context.rotate(Math.PI / 180 * (circles[circle_index].turn_angle + 270));




    /*Big Circle*/
    // Blue color
    context.fillStyle = circles[circle_index].color;
    context.beginPath();
    context.arc(big_circle_x, big_circle_y, 30, 0, Math.PI*2, true);
    context.closePath();
    context.fill();

    /*Triangal Drawing*/
    context.beginPath();
    context.moveTo(tright_corner_x, tright_corner_y);   // Right
    context.lineTo(ttop_corner_x, ttop_corner_y);       // Top
    context.lineTo(tleft_corner_x, tleft_corner_y);     // Left
    context.setLineDash([5, 8])
    context.lineTo(tright_corner_x, tleft_corner_y);    // bottom line

    context.lineWidth="1";
    context.strokeStyle="black";
    context.stroke();
    context.fillStyle = 'rgba(0,0,0,0)';
    context.setLineDash([])
    context.fill();


    var rotatedCord = rotatedAxisCordinates(tright_corner_x, tright_corner_y, circles[circle_index].turn_angle + 270);
    circles[circle_index].triangle_rax = rotatedCord.x + circles[circle_index].x_cordinate;
    circles[circle_index].triangle_ray = rotatedCord.y + circles[circle_index].y_cordinate;


    // Triangle Left
    rotatedCord = rotatedAxisCordinates(tleft_corner_x, tleft_corner_y, circles[circle_index].turn_angle + 270);
    circles[circle_index].triangle_lbx =  rotatedCord.x + circles[circle_index].x_cordinate;
    circles[circle_index].trinagle_lby =  rotatedCord.y + circles[circle_index].y_cordinate;

    // Triangle Top
    rotatedCord = rotatedAxisCordinates(ttop_corner_x, ttop_corner_y, circles[circle_index].turn_angle + 270);
    circles[circle_index].triangle_tkx = rotatedCord.x + circles[circle_index].x_cordinate;
    circles[circle_index].triangle_tky = rotatedCord.y + circles[circle_index].y_cordinate;



    /*Check if enemy is in vision field, if yess activate red light if not set to green*/
    var foundEnemy = circleInVisionField(circle_index, deter_index, circles[deter_index].x_cordinate, circles[deter_index].y_cordinate, 30);

    if(foundEnemy)
    {
        // Enemy found, set to red and be ready!
        fill_style = "#c82124";
    }
    else
    {
        // Enemy lost!
        fill_style = "#228B22";
    }


    /*Small cricle*/
    // Red color
    context.fillStyle = fill_style;
    context.beginPath();
    context.arc(small_circle_x, small_circle_y ,15, 0, Math.PI*2, true);
    context.fill();


    context.restore();


    // Shooting algorithm
    if(circles[circle_index].shooted)
    {
        // Saving after first shoot rotation degree and position
        if(circles[circle_index].projectile_x == 0 && circles[circle_index].projectile_y == 22)
        {
            circles[circle_index].last_degree_before_rotation = circles[circle_index].turn_angle + 270;
            circles[circle_index].pos_before_shotx = circles[circle_index].x_cordinate;
            circles[circle_index].pos_before_shoty = circles[circle_index].y_cordinate;
        }

        context.save();
        context.translate( circles[circle_index].pos_before_shotx, circles[circle_index].pos_before_shoty);
        context.rotate(Math.PI / 180 * circles[circle_index].last_degree_before_rotation);

        context.fillStyle = "#000000";
        context.beginPath();
        context.arc(circles[circle_index].projectile_x, circles[circle_index].projectile_y ,5, 0, Math.PI*2, true);
        context.fill();

        circles[circle_index].projectile_y += 30;
        context.restore();


        rotatedCord = rotatedAxisCordinates(circles[circle_index].projectile_x, circles[circle_index].projectile_y, circles[circle_index].last_degree_before_rotation);

        circles[circle_index].proj_transf_x = rotatedCord.x + circles[circle_index].x_cordinate;
        circles[circle_index].proj_transf_y = rotatedCord.y + circles[circle_index].y_cordinate;



        intersects = circles_intersect(circles[circle_index].proj_transf_x, circles[circle_index].proj_transf_y, circles[deter_index].x_cordinate, circles[deter_index].y_cordinate, 30, 5);
        if(intersects == 1 || intersects == 0)
        {
            //console.log("Intersected!!");
            circles[circle_index].number_of_hits += 30;
            //console.log("Result is: " + circles[0].number_of_hits + " : " + circles[1].number_of_hits);
            circles[circle_index].shooted = false;
            circles[circle_index].shooting_time = 100;
        }
    }
    else
    {
        circles[circle_index].projectile_x = 0;
        circles[circle_index].projectile_y = 22;
    }



    /*Checking if a ball is visible in the field*/
    var visibleBullet = false;
    if(circles[deter_index].shooted)
    {
        visibleBullet = circleInVisionField(circle_index, deter_index,  circles[deter_index].proj_transf_x, circles[deter_index].proj_transf_y, 5);
        if(visibleBullet)
        {
            // Enemy found, set to red and be ready!
            //console.log("Found enemy bullet!!");
        }
        else
        {
            // Enemy lost!
            //console.log("Nope");
        }
    }


    updateDataView(circles[circle_index].x_cordinate, circles[circle_index].y_cordinate, degrees, foundEnemy,
        visibleBullet, circles[circle_index].shooting_time, circle_index, circles[circle_index].number_of_hits, circles[deter_index].number_of_hits);


    /*Draw two lines*/
    context.moveTo(650,0);
    context.lineTo(650,800);
    context.stroke();
}



function updateDataView(cordx, cordy, degrees, seeEnemy, seeBullet, shootingTime, circle_num, first_hits, second_hits)
{
    if(circle_num == 1)
        circle_name = "first";
    else if (circle_num == 0)
        circle_name = "second";

    document.getElementById("cordinates-"+ circle_name).innerHTML = "(" + Math.round(cordx) + ":" + Math.round(cordy) + ")";
    document.getElementById("degress-"+ circle_name).innerHTML = Math.round(degrees * 100) / 100;
    document.getElementById("enemy-detected-" + circle_name).innerHTML = seeEnemy;
    document.getElementById("bullet-detected-" + circle_name).innerHTML = seeBullet;
    document.getElementById("shooting-time-" + circle_name).innerHTML = shootingTime;
    document.getElementById("result").innerHTML = second_hits + ":" + first_hits;
}




function circleInVisionField(index, enemy_index, target_x, target_y, circle_r)
{
    first_line = detectCollision(circles[index].triangle_tkx,  circles[index].triangle_tky,  circles[index].triangle_lbx, circles[index].trinagle_lby, target_x, target_y, circle_r);
    second_line = detectCollision(circles[index].triangle_tkx,  circles[index].triangle_tky,  circles[index].triangle_rax, circles[index].triangle_ray, target_x, target_y, circle_r);

    if(first_line == false && second_line == true)
    {
        return true;
    }
    else
    {
        return false;
    }
}



// Find when bullet hits circle
function circles_intersect(x1, y1, x2, y2, r1, r2)
{
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    radSumSq = (r1 + r2) * (r1 + r2);
    if (distSq == radSumSq)
        return 1;
    else if (distSq > radSumSq)
        return -1;
    else
        return 0;
}


// Finds next point for bullet given x cordinate
function find_next_point(x, m, b)
{
    return m * x + b;
}

// Calculates bullet linear equation
function find_line_equation(x1, y1, x2, y2, index)
{
    m = (y2 - y1) / (x2 - x1);
    b = y1 - m * x1;
    circles[index].projectile_eqm = m;
    circles[index].projectile_eqb = b;
}


function rotatedAxisCordinates(x, y, degree)
{

    transf_cordinate = new Vector();
    transf_cordinate.x = x * Math.cos(Math.radians(degree)) - y * Math.sin(Math.radians(degree));
    transf_cordinate.y = y * Math.cos(Math.radians(degree)) + x * Math.sin(Math.radians(degree));

    return transf_cordinate;
}

Math.radians = function(degrees) {
    return degrees * Math.PI / 180;
};


function setWeights()
{

    circle_index = 1;
    if (circle_two_keys["w_key"]) {
        circles[circle_index].moving_mode = 1;
    }
    else if(circle_two_keys["s_key"])
    {
        circles[circle_index].moving_mode = -1;
    }
    else{
        circles[circle_index].moving_mode = 0;
    }

    // Right
    if (circle_two_keys["a_key"]) {
        circles[circle_index].turn_angle += 5;
    }

    // LEFT
    if (circle_two_keys["d_key"]) {
        circles[circle_index].turn_angle += -5;
    }

    // Right
    if (circle_two_keys["p_key"]) {
        if(circles[circle_index].current_vision < 400)
            circles[circle_index].current_vision += 8;
    }

    // LEFT
    if (circle_two_keys["o_key"]) {
        if(circles[circle_index].current_vision > 70)
            circles[circle_index].current_vision += -8;
    }




    /* Deciding if shoot or not */
    if(circle_two_keys["spacebar_key"] && (circles[circle_index].shooted == false))
    {


        circles[circle_index].shooted = true;

    }
    else if(circles[circle_index].shooting_time < 0)
    {
        circles[circle_index].shooted = false;
        circles[circle_index].shooting_time = 100;

    }
    else if(circles[circle_index].shooting_time > -1 && (circles[circle_index].shooted == true))
    {
        circles[circle_index].shooting_time -= 1;

    }


    /*Rectangle keys up*/
    // tank_one_mode = 0;
    // UP
    circle_index = 0;
    if (circle_one_keys["up_key"]) {
        circles[circle_index].moving_mode = 1;
    }
    else if(circle_one_keys["down_key"])
    {
        circles[circle_index].moving_mode = -1;
    }
    else{
        circles[circle_index].moving_mode = 0;
    }

    // Right
    if (circle_one_keys["right_key"]) {
        circles[circle_index].turn_angle += 5;
    }

    // LEFT
    if (circle_one_keys["left_key"]) {
        circles[circle_index].turn_angle += -5;
    }

    // Right
    if (circle_one_keys["z_key"]) {
        if(circles[circle_index].current_vision < 400)
            circles[circle_index].current_vision += 8;
    }

    // LEFT
    if (circle_one_keys["x_key"]) {
        if(circles[circle_index].current_vision > 70)
            circles[circle_index].current_vision += -8;
    }


    //console.log("WTF: " + circle_one_keys["one_key"])

    /*Deciding if shoot or number_of_hits*/
    if(circle_one_keys["one_key"] && (circles[circle_index].shooted == false))
    {
        circles[circle_index].shooted = true;


    }
    else if(circles[circle_index].shooting_time < 0)
    {
        circles[circle_index].shooted = false;
        circles[circle_index].shooting_time = 100;

    }
    else if(circles[circle_index].shooting_time > -1 && (circles[circle_index].shooted == true))
    {
        circles[circle_index].shooting_time -= 1;

    }

}


/*If key has been stoped press set to false*/
function keyup_handler(event) {
    //console.log("Key up: " + event.keyCode);


    /*Tank keys up*/
    // tank_two_mode = 0;
    // W => 1
    if (event.keyCode == 87) {
        circle_two_keys["w_key"] = false;
    }

    // S => -1
    if (event.keyCode == 83) {
        circle_two_keys["s_key"] = false;
    }

    // D => -=5
    if (event.keyCode == 65) {
        circle_two_keys["d_key"] = false;
    }

    // A => +=5
    if (event.keyCode == 68) {
        circle_two_keys["a_key"] = false;
    }

    if(event.keyCode == 79)
    {
        circle_two_keys["o_key"] = false;
    }

    if(event.keyCode == 80)
    {
        circle_two_keys["p_key"] = false;
    }

    if(event.keyCode == 32)
    {
        circle_two_keys["spacebar_key"] = false;
    }

    /*Rectangle keys up*/
    // tank_one_mode = 0;
    // UP
    if (event.keyCode == 38) {
        circle_one_keys["up_key"] = false;
    }

    // DOWN
    if (event.keyCode == 40) {
        circle_one_keys["down_key"] = false;
    }

    // Right
    if (event.keyCode == 39) {
        circle_one_keys["right_key"] = false;
    }

    // LEFT
    if (event.keyCode == 37) {
       circle_one_keys["left_key"] = false;
    }

    if(event.keyCode == 90)
    {
        circle_one_keys["z_key"] = false;
    }

    if(event.keyCode == 88)
    {
        circle_one_keys["x_key"] = false;
    }

    if(event.keyCode == 97)
    {
        circle_one_keys["one_key"] = false;
    }


}



/*If key has been pressed, set it to true*/
$(document.body).keydown(function(event) {
    //console.log("Key down: " + event.keyCode);

    // W => 1
   if (event.keyCode == 87) {
        circle_two_keys["w_key"] = true;
    }

    // S => -1
    if (event.keyCode == 83) {
        circle_two_keys["s_key"] = true;
    }

    // D => -=5
    if (event.keyCode == 65) {
        circle_two_keys["d_key"] = true;
    }

    // A => +=5
    if (event.keyCode == 68) {
        circle_two_keys["a_key"] = true;
    }

    if(event.keyCode == 79)
    {
        circle_two_keys["o_key"] = true;
    }

    if(event.keyCode == 80)
    {
        circle_two_keys["p_key"] = true;
    }

    if(event.keyCode == 32)
    {
        circle_two_keys["spacebar_key"] = true;
    }





    // UP
    if (event.keyCode == 38) {
        circle_one_keys["up_key"] = true;
    }

    // DOWN
    if (event.keyCode == 40) {
        circle_one_keys["down_key"] = true;
    }

    // Right
    if (event.keyCode == 39) {
        circle_one_keys["right_key"] = true;
    }

    // LEFT
    if (event.keyCode == 37) {
       circle_one_keys["left_key"] = true;
    }

    if(event.keyCode == 90)
    {
        circle_one_keys["z_key"] = true;
    }

    if(event.keyCode == 88)
    {
        circle_one_keys["x_key"] = true;
    }

    if(event.keyCode == 97)
    {
        circle_one_keys["one_key"] = true;
    }

});


function detectCollision(x1, y1, x2, y2,  cx, cy, cr)
{

    d = (cx - x1) * (y2 - y1) - (cy -y1) * (x2 - x1);
    return d < 0;
}


