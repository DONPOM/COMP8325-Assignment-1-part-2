
<!-- Original:  Robert Bui (astrogate@hotmail.com) -->
<!-- Web Site:  http://astrogate.virtualave.net -->

<!-- This script and many more are available free online at -->
<!-- The JavaScript Source!! http://javascript.internet.com -->

<!-- Begin
var interval = 3; // delay between rotating images (in seconds)
var random_display = 0; // 0 = no, 1 = yes
interval *= 1000;

var image_index = 0;
image_list = new Array();
image_list[image_index++] = new imageItem("http://www.ddmotorsystems.com/slideshow/Wheelie.jpg");
image_list[image_index++] = new imageItem("http://www.ddmotorsystems.com/slideshow/BlueMonster.jpg");
image_list[image_index++] = new imageItem("http://www.ddmotorsystems.com/slideshow/ProEVCart.jpg");
image_list[image_index++] = new imageItem("http://www.ddmotorsystems.com/slideshow/KrazyKartz.jpg");
image_list[image_index++] = new imageItem("http://www.ddmotorsystems.com/slideshow/6Pass.jpg");
var number_of_image = image_list.length;
function imageItem(image_location) {
this.image_item = new Image();
this.image_item.src = image_location;
}
function get_ImageItemLocation(imageObj) {
return(imageObj.image_item.src)
}
function generate(x, y) {
var range = y - x + 1;
return Math.floor(Math.random() * range) + x;
}
function getNextImage() {
if (random_display) {
image_index = generate(0, number_of_image-1);
}
else {
image_index = (image_index+1) % number_of_image;
}
var new_image = get_ImageItemLocation(image_list[image_index]);
return(new_image);
}
function rotateImage(place) {
var new_image = getNextImage();
document[place].src = new_image;
var recur_call = "rotateImage('"+place+"')";
setTimeout(recur_call, interval);
}
//  End -->
