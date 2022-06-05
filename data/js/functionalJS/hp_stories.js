/* FILE USED FOR HPBANNERSTORIES */
var $j = jQuery;

function swapem() {
 /* Rotates the stories automatically using setInterval to call function */
var getnext = $j("#divhold .active").next();
var current = $j("#divhold .active");
	if (current.is(':last-child')) {
       var getnext = $j("#divhold div:first-child");
    }
	/* Rotates the buttons */ 
var getnextb = $j("#hp_dots .highlight").next();
var currentb = $j("#hp_dots .highlight");
	if (currentb.is(':last-child')) {
       var getnextb = $j('#hp_dots div:first-child');
    }

/* Fades the divs inside divhold */
    $j("#divhold div").fadeTo("slow", "0.3", function() {
        $j("#divhold div").removeClass("active");
        getnext.fadeTo("slow", "1").addClass("active")
    });
/* Animate the buttons according to the div that's active*/
       $j("#hp_dots div").removeClass("highlight");
           getnextb.addClass("highlight");
}


$j(document).ready(function() {
  
/*Set Interval to swap the images and buttons automatically */
 
var divswap = setInterval('swapem()', 5000);
$j("#hp_learnmore").click(function() { clearInterval(divswap); });
/*click on the dot and div becomes active */
    $j('body').click(function(event) {
        if ($j(event.target).is("#dot1")) {
            $j(event.target).toggleClass('highlight');
            $j("#dot2").removeClass('highlight'); $j("#dot3").removeClass('highlight'); $j("#dot4").removeClass('highlight');
            if ($j(".cbox1").is("active")) {
                return;
            } else {
                $j("#divhold div").fadeTo("slow", "0.3", function() {
                    $j("#divhold div").removeClass("active");
                    $j(".cbox1").fadeTo("slow", "1").addClass("active")
                });
            }
        }

        if ($j(event.target).is("#dot2")) {
            $j(event.target).toggleClass('highlight');
            $j("#dot1").removeClass('highlight'); $j("#dot3").removeClass('highlight'); $j("#dot4").removeClass('highlight');


            $j("#divhold div").fadeTo("slow", "0.3", function() {
                $j("#divhold div").removeClass("active");
                $j(".cbox2").fadeTo("slow", "1").addClass("active")
            });
        }

        if ($j(event.target).is("#dot3")) {
            $j(event.target).toggleClass('highlight');
            $j("#dot1").removeClass('highlight'); $j("#dot2").removeClass('highlight'); $j("#dot4").removeClass('highlight');

            $j("#divhold div").fadeTo("slow", "0.3", function() {
                $j("#divhold div").removeClass("active");
                $j(".cbox3").fadeTo("slow", "1").addClass("active")
            });
        }

        if ($j(event.target).is("#dot4")) {
            $j(event.target).toggleClass('highlight');
            $j("#dot1").removeClass('highlight'); $j("#dot2").removeClass('highlight'); $j("#dot3").removeClass('highlight');

            $j("#divhold div").fadeTo("slow", "0.3", function() {
                $j("#divhold div").removeClass("active");
                $j(".cbox4").fadeTo("slow", "1").addClass("active")
            });
        }
        
       
        

    });



 	
    for (k = 1; k < 5; k++) {
       /* $j('#dot' + j).hover(function() { $j(this).addClass("highlight"); }, function() { $j(this).removeClass("highlight"); });*/
        $j('#dot' + k).click(function() { clearInterval(divswap); });
       
    }
}
);
