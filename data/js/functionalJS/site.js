"use strict";

/*  ----------------------------------------------------------------------------
    td_detect - browser detection object (instance)
    v1.1
 */

var td_detect = new function () {

    //constructor
    this.is_ie8 = false;
    this.is_ie9 = false;
    this.is_ie10 = false;
    this.is_ie11 = false;
    this.is_ie = false;
    this.is_safary = false;
    this.is_chrome = false;
    this.is_ipad = false;

    this.is_touch_device = false;

    this.has_history = false;

    this.is_phone_screen = false;

    this.is_ios = false;

    this.is_android = false;

    // is touch device ?
    this.is_touch_device = !!('ontouchstart' in window);

    this.is_mobile_device = false;

    // detect ie8
    if (jQuery('html').is('.ie8')) {
        this.is_ie8 = true;
        this.is_ie = true;
    }



    // detect ie9
    if (jQuery('html').is('.ie9')) {
        this.is_ie9 = true;
        this.is_ie = true;
    }

    // detect ie10 - also adds the ie10 class //it also detects windows mobile IE as IE10
    if (navigator.userAgent.indexOf("MSIE 10.0") > - 1) {
        jQuery("html").addClass("ie10");
        this.is_ie10 = true;
        this.is_ie = true;
        //alert('10');
    }

    //ie 11 check - also adds the ie11 class - it may detect ie on windows mobile
    if (!!navigator.userAgent.match(/Trident.*rv\:11\./)) {
        jQuery("html").addClass("ie11");
        this.is_ie11 = true;
        //this.is_ie = true; //do not flag ie11 as is_ie
        //alert('11');
    }


    //do we have html5 history support?
    if (window.history && window.history.pushState) {
        this.has_history = true;
    }

    //check for safary
    if (navigator.userAgent.indexOf('Safari') != - 1 && navigator.userAgent.indexOf('Chrome') == - 1) {
        this.is_safary = true;
    }

    //chrome and chrome-ium check
    this.is_chrome = /chrom(e|ium)/.test(navigator.userAgent.toLowerCase());

    this.is_ipad = navigator.userAgent.match(/iPad/i) != null;



    this.is_ios = /(iPad|iPhone|iPod)/g.test( navigator.userAgent );


    //detect if we run on a mobile device - ipad included - used by the modal / scroll to @see scroll_into_view
    if ( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
        this.is_mobile_device = true;
    }

    /**
         * function to check the phone screen
         * @see td_events
         * The jQuery windows width is not reliable cross browser!
         */
    this.run_is_phone_screen = function () {
        if ((jQuery(window).width() < 768 || jQuery(window).height() < 768) && this.is_ipad === false) {
            this.is_phone_screen = true;

        } else {
            this.is_phone_screen = false;
        }

        //console.log(this.is_phone_screen + ' ' + jQuery(window).width() + ' ' + jQuery(window).height());
    }



    this.run_is_phone_screen();


    //test for android
    var user_agent = navigator.userAgent.toLowerCase();
    if (user_agent.indexOf("android") > - 1) {
        this.is_android = true;
    }



};




/*  ----------------------------------------------------------------------------
    tagDiv magic cache
    v1.0
 */
var td_local_cache = {
    data: {},
    remove: function (resurce_id) {
        delete td_local_cache.data[resurce_id];
    },
    exist: function (resurce_id) {
        return td_local_cache.data.hasOwnProperty(resurce_id) && td_local_cache.data[resurce_id] !== null;
    },
    get: function (resurce_id) {
        return td_local_cache.data[resurce_id];
    },
    set: function (resurce_id, cachedData) {
        td_local_cache.remove(resurce_id);
        td_local_cache.data[resurce_id] = cachedData;
    }
};

/*
    td_util.js
    v1.1
 */

"use strict";



/*  ----------------------------------------------------------------------------
    tagDiv utility class
 */
var td_util = {


    /**
         * utility function, used by td_post_images.js
         * @param class_selector
         */
    image_move_class_to_figure: function (class_selector) {
        jQuery('figure .' + class_selector).each(function() {
            jQuery(this).parent().parent().addClass(class_selector);
            jQuery(this).removeClass(class_selector);
        });
    },



    /**
         * safe function to read variables passed by the theme via the js buffer. If by some kind of error the variable is missing from the global scope, this function will return false
         * @param variable_name
         * @returns {*}
         */
    get_backend_var: function(variable_name) {
        if (typeof window[variable_name] === 'undefined') {
            return '';
        }
        return window[variable_name];
    },



    /**
         * scrolls to a dom element
         * @param dom_element
         */
    scroll_to_element: function(dom_element, duration) {
        td_is_scrolling_animation = true;
        jQuery("html, body").stop();


        var dest;

        //calculate destination place
        if (dom_element.offset().top > jQuery(document).height() - jQuery(window).height()) {
            dest = jQuery(document).height() - jQuery(window).height();
        } else {
            dest = dom_element.offset().top;
        }
        //go to destination
        jQuery("html, body").animate({
            scrollTop: dest 
        }, {
            duration: duration,
            easing: 'easeInOutQuart',
            complete: function() {
                td_is_scrolling_animation = false;
            }
        });
    },


    /**
         * scrolls to a dom element - the element will be close to the center of the screen
         * !!! compensates for long distances !!!
         */
    scroll_into_view: function (dom_element) {

        if (td_detect.is_mobile_device === true) {
            return; //do not run on any mobile device
        }

        td_is_scrolling_animation = true;
        jQuery("html, body").stop();


        var destination = dom_element.offset().top;
        destination = destination - 150;

        var distance = Math.abs(jQuery(window).scrollTop() - destination);
        var computed_time = distance / 5;
        //console.log(distance + ' -> ' + computed_time +  ' -> ' + (1100+computed_time));

        //go to destination
        jQuery("html, body").animate({
            scrollTop: destination 
        }, {
            duration: 1100 + computed_time,
            easing: 'easeInOutQuart',
            complete: function() {
                td_is_scrolling_animation = false;
            }
        });
    },

    /**
         * scrolls to a position
         * @param px_from_top - pixels from top
         */
    scroll_to_position: function(px_from_top, duration) {
        td_is_scrolling_animation = true;
        jQuery("html, body").stop();

        //go to destination
        jQuery("html, body").animate({
            scrollTop: px_from_top 
        }, {
            duration: duration,
            easing: 'easeInOutQuart',
            complete: function() {
                td_is_scrolling_animation = false;
            }
        });
    },
    td_move_y: function td_move_Y (elm, value) {
        var translate = 'translate3d(0px,' + value + 'px, 0px)';
        elm.style['-webkit-transform'] = translate;
        elm.style['-moz-transform'] = translate;
        elm.style['-ms-transform'] = translate;
        elm.style['-o-transform'] = translate;
        elm.style.transform = translate;
    }


};


/**
 * Created by ra on 6/27/14.
 * copyright tagDiv 2014
 */


var td_affix = {

    //settings, obtained from ext
    menu_selector: '',
    //the affix menu (this element will get the td-affix)
    menu_wrap_selector: '',
    //the menu wrapper / placeholder
    tds_snap_menu: '',
    //the panel setting


    top_offset: 0,
    //how much the menu is moved from the original position when it's affixed
    menu_offset: 0,
    //used to hide the menu on scroll
    is_requestAnimationFrame_running: false,
    //prevent multiple calls to requestAnimationFrame
    is_menu_affix: false,
    //the current state of the menu, true if the menu is affix
    is_top_menu: false,
    //true when the menu is at the top of the screen (0px topScroll)

    //menu offset boundaries - so we do not fire the animation event when the boundary is hit
    menu_offset_max_hit: false,
    menu_offset_min_hit: true,


    scroll_window_scrollTop_last: 0,
    //last scrollTop position, used to calculate the scroll direction

    /**
         * run the affix, we use the menu wrap selector to compute the menu position from top
         *
           {
                menu_selector: '.td-header-main-menu',
                menu_wrap_selector: '.td-header-menu-wrap',
                tds_snap_menu: td_util.get_backend_var('tds_snap_menu')
            }
         */
    init : function init (atts) {

        //read the settings
        td_affix.menu_selector = atts.menu_selector;
        td_affix.menu_wrap_selector = atts.menu_wrap_selector;
        td_affix.tds_snap_menu = atts.tds_snap_menu;

        //the snap menu is disabled from the panel
        if (!td_affix.tds_snap_menu) {
            return;
        }

        //compute on semi dom ready
        td_affix.compute_top();


        //recompute when all the page + logos are loaded
        jQuery(window).load(function() {
            td_affix.compute_top();

            //recompute after 1 sec for retarded phones
            setTimeout(function() {
                td_affix.compute_top();
            }, 1000);
        });




    },

    /**
         * called by td_events.js on scroll
         */
    td_events_scroll: function td_events_scroll(scrollTop) {

        //do not run if we don't have a snap menu
        if (!td_affix.tds_snap_menu) {
            return;
        }


        /*  ----------------------------------------------------------------------------
                    scroll direction + delta (used by affix for now)
                    to run thios code:
                     - td_affix.tds_snap_menu != '' (from above)
                     - td_affix.tds_snap_menu != 'snap'
                 */
        if (td_affix.tds_snap_menu != 'snap') {
            //do not run on snap
            if ((td_affix.tds_snap_menu != 'smart_snap_mobile' || td_detect.is_phone_screen === true)) {
                // different from smart_snap_mobile or td_detect.is_phone_screen === true
                //console.log('rrr');
                var scroll_direction = '';
                var scrollDelta = 0;

                //check the direction
                if (scrollTop != td_affix.scroll_window_scrollTop_last) {
                    //compute direction only if we have different last scroll top
                    // compute the direction of the scroll
                    if (scrollTop > td_affix.scroll_window_scrollTop_last) {
                        scroll_direction = 'down';
                    } else {
                        scroll_direction = 'up';
                    }
                    //calculate the scroll delta
                    scrollDelta = Math.abs(scrollTop - td_affix.scroll_window_scrollTop_last);
                }

                td_affix.scroll_window_scrollTop_last = scrollTop;
            }
        }

        /*  ---------------------------------------------------------------------------- */



        //if the menu is in the affix state
        if (scrollTop > td_affix.top_offset || td_affix.is_top_menu === true) {

            //get the menu element
            var td_affix_menu_element = jQuery(td_affix.menu_selector);

            //turn affix on for it
            td_affix._affix_on(td_affix_menu_element);


            //if the menu is only with snap or we are on smart_snap_mobile + mobile, our job here in this function is done, return
            if (td_affix.tds_snap_menu == 'snap' || (td_affix.tds_snap_menu == 'smart_snap_mobile' && td_detect.is_phone_screen === false)) {
                return;
            }

            /*    ---  end simple snap  ---  */


            /*  ----------------------------------------------------------------------------
                            check scroll directions (we may also have scroll_direction = '', that's why we have to check for the specific state (up or down))
                         */



            // boundary check - to not run the position on each scroll event
            if ((td_affix.menu_offset_max_hit === false && scroll_direction == 'down') || (td_affix.menu_offset_min_hit === false && scroll_direction == 'up')) {
                //request animation frame
                if (td_affix.is_requestAnimationFrame_running === false) {
                    window.requestAnimationFrame(function() {

                        var offset = 0;

                        if (scroll_direction == 'down') {
                            //compute the offset
                            offset = td_affix.menu_offset - scrollDelta;
                            if (offset < - 80) {
                                offset = - 80;
                            }
                        } else if (scroll_direction == 'up') {
                            //compute the offset
                            offset = td_affix.menu_offset + scrollDelta;
                            if (offset > 0) {
                                offset = 0;
                            }
                        }


                        td_affix.is_requestAnimationFrame_running = true;

                        //move the menu
                        td_util.td_move_y(td_affix_menu_element[0], offset);
                        //td_affix_menu_element.css({top: (offset) + 'px'});  //legacy menu move code

                        //check boundaries
                        if (offset == 0) {
                            td_affix.menu_offset_min_hit = true;
                        } else {
                            td_affix.menu_offset_min_hit = false;
                        }

                        if (offset == - 80) {
                            td_affix.menu_offset_max_hit = true;
                            //also hide the menu when it's 100% out of view on ios - the safari header is transparent and we can see the menu
                            if (td_detect.is_ios === true) {
                                td_affix_menu_element.hide();
                            }
                        } else {
                            td_affix.menu_offset_max_hit = false;

                            if (td_detect.is_ios === true) {
                                //ios safari fix
                                td_affix_menu_element.show();
                            }
                        }


                        td_affix.is_requestAnimationFrame_running = false;




                        td_affix.menu_offset = offset; //update the current offset of the menu


                    }, td_affix_menu_element[0]);

                }
                //console.log(offset + ' ' + scroll_direction);

            }
            //end boundary check



        } else {
            td_affix._affix_off(jQuery(td_affix.menu_selector));
        }

    },


    /**
         * calculates the affix point (the distance from the top when affix should be enabled)
         * @see td_affix.init()
         * @see td_events
         */
    compute_top: function compute_top() {
        td_affix.top_offset = jQuery(td_affix.menu_wrap_selector).offset().top;

        //check to see if the menu is at the top of the screen
        if (td_affix.top_offset == 1) {
            //switch to affix - because the menu is at the top of the page
            //td_affix._affix_on(jQuery(td_affix.menu_selector));
            td_affix.is_top_menu = true;
        } else {
            //check to see the current top offset
            td_affix.is_top_menu = false;

        }
        td_affix.td_events_scroll(jQuery(window).scrollTop());

        //alert(td_affix.top_offset);
        //console.log('computed: ' + td_affix.top_offset);
    },



    /**
         * turns affix on for the menu element
         * @param td_affix_menu_element
         * @private
         */
    _affix_on: function _affix_on(td_affix_menu_element) {
        if (td_affix.is_menu_affix === false) {
            //make the menu fixed
            td_affix_menu_element.addClass('td-affix');

            //add body-td-affix class on body for header style 8 -> when scrolling down the window jumps 76px up when the menu is changing from header style 8 default to header style 8 affix
            jQuery('body').addClass('body-td-affix');

            td_affix.is_menu_affix = true;
        }
    },



    /**
         * Turns affix off for the menu element
         * @param td_affix_menu_element
         * @private
         */
    _affix_off: function _affix_on(td_affix_menu_element) {
        if (td_affix.is_menu_affix === true) {
            //make the menu normal
            jQuery(td_affix.menu_selector).removeClass('td-affix');

            //remove body-td-affix class on body for header style 8 -> when scrolling down the window jumps 76px up when the menu is changing from header style 8 default to header style 8 affix
            jQuery('body').removeClass('body-td-affix');

            td_affix.is_menu_affix = false;

            //move the menu to 0 (ios seems to skip animation frames)
            td_util.td_move_y(td_affix_menu_element[0], 0);

            if (td_detect.is_ios === true) {
                td_affix_menu_element.show();
            }

        }
    }



}



/*  ----------------------------------------------------------------------------
    Affix menu
 */
td_affix.init({
    menu_selector: '.td-menu-background',
    menu_wrap_selector: '.td-menu-placeholder',
    tds_snap_menu: td_util.get_backend_var('tds_snap_menu')
});




/*
    tagDiv - 2014 - Newspaper WordPress theme v4.0
    Our portfolio:  http://themeforest.net/user/tagDiv/portfolio

    Thanks for using our theme! :)
*/





var td_is_slide_moving = false; //disable touch when the touch sliders are moving




/*  ----------------------------------------------------------------------------
 Menu script
 */

jQuery('#td-top-menu .sf-menu').supersubs({
    minWidth: 10,
    // minimum width of sub-menus in em units
    maxWidth: 40,
    // maximum width of sub-menus in em units
    extraWidth: 1 // extra width can ensure lines don't sometimes turn over
});

if (td_detect.is_touch_device) {
    //touch
    jQuery('#td-top-menu .sf-menu').superfish({
        delay: 300,
        speed: 'fast',
        useClick: true
    });
} else {
    //not touch
    jQuery('#td-top-menu .sf-menu').superfish({
        delay: 500,
        speed: 200,
        useClick: false
    });
}

/*  ----------------------------------------------------------------------------
    On load
 */
jQuery().ready(function jQuery_ready() {



    //resize all the videos if we have them
    td_resize_videos();









    //put focus on search box in blog header
    jQuery('#search-button').click(function() {
        jQuery(this).delay(200).queue(function() {
            document.getElementById("td-header-search").focus();
            jQuery(this).dequeue();
        });
    });


    //retina images
    td_retina();

    //colorbox
    jQuery('.td-featured-img').colorbox({
        maxWidth: "95%",
        maxHeight: "95%",
        fixed: true
    });





    td_ajax_search();


    //srun the mobile menu on phones and on desktop (no ipad or big tablets with touch)
    if (td_detect.is_phone_screen || td_detect.is_touch_device === false) {
        //alert('mobile menu');
        td_mobile_menu2();
    }







    td_fake_clicks();







    //fake placeholder for ie
    jQuery('input, textarea').placeholder();



    //more stories box
    td_more_articles_box.init();



    if (td_detect.is_chrome === true || td_detect.is_ie10 || td_detect.is_ie11) {
        td_smooth_scroll();
    }


    setTimeout(function() {
        td_resize_page_sliders()
    }, 1500);

}); //end on load


/**
 * windlow.load
 */
jQuery(window).load(function() {
    td_resize_page_sliders();
});



/*  ----------------------------------------------------------------------------
 Scroll to top + animation stop
 */

var td_is_scrolling_animation = false;
var td_mouse_wheel_or_touch_moved = false; //we want to know if the user stopped the animation via touch or mouse move

//stop the animation on mouse wheel
jQuery(document).bind('mousewheel DOMMouseScroll MozMousePixelScroll', function(e) {
    if (td_is_scrolling_animation === false) {
        return;
    } else {
        td_is_scrolling_animation = false;
        td_mouse_wheel_or_touch_moved = true;

        jQuery("html, body").stop();
    }
});

//stop the animation on touch
if (document.addEventListener) {
    document.addEventListener('touchmove', function(e) {
        if (td_is_scrolling_animation === false) {
            return;
        } else {
            td_is_scrolling_animation = false;
            td_mouse_wheel_or_touch_moved = true;
            jQuery("html, body").stop();
        }
    }, false);
}

/**
 * called by td_events.js on scroll - back to top
 */
var td_scroll_to_top_is_visible = false;
function td_events_scroll_scroll_to_top(scrollTop) {
    if (td_is_scrolling_animation) {
        //do not fire the event on animations
        return;
    }
    if (scrollTop > 400) {
        if (td_scroll_to_top_is_visible === false) {
            //only add class if needed
            td_scroll_to_top_is_visible = true;
            jQuery('.td-scroll-up').addClass('td-scroll-up-visible');
        }
    } else {
        if (td_scroll_to_top_is_visible === true) {
            //only add class if needed
            td_scroll_to_top_is_visible = false;
            jQuery('.td-scroll-up').removeClass('td-scroll-up-visible');
        }
    }
}


jQuery('.td-scroll-up').click(function() {
    if (td_is_scrolling_animation) {
        //double check - because when we remove the class, the button is still visible for a while
        return;
    }

    //hide the button
    td_scroll_to_top_is_visible = false;
    jQuery('.td-scroll-up').removeClass('td-scroll-up-visible');

    //hide more articles box
    td_more_articles_box.is_box_visible = false;
    jQuery('.td-more-articles-box').removeClass('td-front-end-display-block');

    //scroll to top
    td_util.scroll_to_position(0, 1200);

    return false;
});


/**
 * More stories box
 */
var td_more_articles_box = {
    is_box_visible: false,
    cookie: '',
    distance_from_top: 400,

    init: function init() {


        //read the cookie
        td_more_articles_box.cookie = td_read_site_cookie('td-cookie-more-articles');


        //setting distance from the top
        if (!isNaN(parseInt(tds_more_articles_on_post_pages_distance_from_top)) && isFinite(tds_more_articles_on_post_pages_distance_from_top) && parseInt(tds_more_articles_on_post_pages_distance_from_top) > 0) {
            td_more_articles_box.distance_from_top = parseInt(tds_more_articles_on_post_pages_distance_from_top);
        } else {
            td_more_articles_box.distance_from_top = 400;
        }

        //adding event to hide the box
        jQuery('.td-close-more-articles-box').click(function() {

            //hiding the box
            jQuery('.td-more-articles-box').removeClass('td-front-end-display-block');
            jQuery('.td-more-articles-box').hide();

            //cookie life
            if (!isNaN(parseInt(tds_more_articles_on_post_time_to_wait)) && isFinite(tds_more_articles_on_post_time_to_wait)) {
                //setting cookie
                td_create_cookie('td-cookie-more-articles', 'hide-more-articles-box', parseInt(tds_more_articles_on_post_time_to_wait));
            }
        });
    },

    /**
         * called by td_events.js on scroll
         */
    td_events_scroll: function td_events_scroll(scrollTop) {

        if (td_is_scrolling_animation) {
            //do not fire the event on animations
            return;
        }

        //check to see if it's enable form panel and also from cookie
        if (td_util.get_backend_var('tds_more_articles_on_post_enable') == "show" && td_more_articles_box.cookie != 'hide-more-articles-box') {

            if (scrollTop > td_more_articles_box.distance_from_top ) {
                if (td_more_articles_box.is_box_visible === false) {
                    jQuery('.td-more-articles-box').addClass('td-front-end-display-block');
                    td_more_articles_box.is_box_visible = true;
                }
            } else {
                if (td_more_articles_box.is_box_visible === true) {
                    jQuery('.td-more-articles-box').removeClass('td-front-end-display-block');
                    td_more_articles_box.is_box_visible = false;
                }
            }
        }
    }
};



//click on a div -> go to a url
function td_fake_clicks() {
    jQuery('.td-fake-click').click(function() {
        window.location = jQuery(this).data("fake-click");
    });
}




var td_resize_timer_id;
jQuery(window).resize(function() {
    clearTimeout(td_resize_timer_id);
    td_resize_timer_id = setTimeout(td_done_resizing, 500);

});

function td_done_resizing() {
    td_resize_videos();
}



/*  ----------------------------------------------------------------------------
    Resize the videos
 */
function td_resize_videos() {
    //youtube in content
    jQuery(document).find('iframe[src*="youtube.com"]').each(function() {

        if (jQuery(this).parent().hasClass("td_wrapper_playlist_player_youtube")) {
            //do nothing for playlist player youtube
        } else {
            var td_video = jQuery(this);
            td_video.attr('width', '100%');
            var td_video_width = td_video.width();
            td_video.css('height', td_video_width * 0.6, 'important');
        }
    });


    //vimeo in content
    jQuery(document).find('iframe[src*="vimeo.com"]').each(function() {

        if (jQuery(this).parent().hasClass("td_wrapper_playlist_player_vimeo")) {
            //do nothing for playlist player vimeo
        } else {
            var td_video = jQuery(this);
            td_video.attr('width', '100%');
            var td_video_width = td_video.width();
            td_video.css('height', td_video_width * 0.6, 'important');
        }
    })

    //daily motion in content
    jQuery(document).find('iframe[src*="dailymotion.com"]').each(function() {
        var td_video = jQuery(this);
        td_video.attr('width', '100%');
        var td_video_width = td_video.width();
        td_video.css('height', td_video_width * 0.6, 'important');
    })
}




/*  ----------------------------------------------------------------------------
    Ajax search
 */
var td_aj_search_cur_sel = 0;
var td_aj_search_results = 0;
var td_aj_first_down_up = true;
function td_ajax_search() {




    jQuery('#td-header-search').keydown(function(event) {

        //console.log(event.keyCode);


        if ((event.which && event.which == 39) || (event.keyCode && event.keyCode == 39) || (event.which && event.which == 37) || (event.keyCode && event.keyCode == 37)) {
            //do nothing on left and right arrows
            td_aj_search_input_focus();
            return;
        }

        if ((event.which && event.which == 13) || (event.keyCode && event.keyCode == 13)) {

            //redirectSearch('q');
            var td_aj_cur_element = jQuery('.td-aj-cur-element');
            if (td_aj_cur_element.length > 0) {
                //alert('ra');
                var td_go_to_url = td_aj_cur_element.find('.entry-title a').attr('href');
                window.location = td_go_to_url;
            } else {
                jQuery(this).parent().parent().submit();
            }

            return false; //redirect for search on enter
        } else {

            if ((event.which && event.which == 40) || (event.keyCode && event.keyCode == 40)) {
                // down
                td_aj_search_move_prompt_down();
                return false; //disable the envent

            } else if ((event.which && event.which == 38) || (event.keyCode && event.keyCode == 38)) {
                //up
                td_aj_search_move_prompt_up();
                return false; //disable the envent
            } else {

                //for backspace we have to check if the search query is empty and if so, clear the list
                if ((event.which && event.which == 8) || (event.keyCode && event.keyCode == 8)) {
                    //if we have just one character left, that means it will be deleted now and we also have to clear the search results list
                    var search_query = jQuery(this).val();
                    if (search_query.length == 1) {
                        jQuery('#td-aj-search').empty();
                    }

                }

                //various keys
                td_aj_search_input_focus();
                setTimeout("td_ajax_search_do_request()", 100);
            }
            return true;
        }

    });



}

//moves the select up
function td_aj_search_move_prompt_up() {


    if (td_aj_first_down_up === true) {
        td_aj_first_down_up = false;
        if (td_aj_search_cur_sel === 0) {
            td_aj_search_cur_sel = td_aj_search_results - 1;
        } else {
            td_aj_search_cur_sel--;
        }
    } else {
        if (td_aj_search_cur_sel === 0) {
            td_aj_search_cur_sel = td_aj_search_results;
        } else {
            td_aj_search_cur_sel--;
        }
    }


    jQuery('.td_mod_aj_search').removeClass('td-aj-cur-element');



    if (td_aj_search_cur_sel > td_aj_search_results - 1) {
        //the input is selected
        jQuery('.td-search-form').fadeTo(100, 1);
    } else {
        td_aj_search_input_remove_focus();
        jQuery('.td_mod_aj_search').eq(td_aj_search_cur_sel).addClass('td-aj-cur-element');
    }



}

//moves the select prompt down
function td_aj_search_move_prompt_down() {

    if (td_aj_first_down_up === true) {
        td_aj_first_down_up = false;
    } else {
        if (td_aj_search_cur_sel === td_aj_search_results) {
            td_aj_search_cur_sel = 0;
        } else {
            td_aj_search_cur_sel++;
        }
    }


    jQuery('.td_mod_aj_search').removeClass('td-aj-cur-element');

    if (td_aj_search_cur_sel > td_aj_search_results - 1 ) {
        //the input is selected
        jQuery('.td-search-form').fadeTo(100, 1);
    } else {
        td_aj_search_input_remove_focus();
        jQuery('.td_mod_aj_search').eq(td_aj_search_cur_sel).addClass('td-aj-cur-element');
    }


}


// puts the focus on the input box
function td_aj_search_input_focus() {
    td_aj_search_cur_sel = 0;
    td_aj_first_down_up = true;
    jQuery('.td-search-form').fadeTo(100, 1);
    jQuery('.td_mod_aj_search').removeClass('td-aj-cur-element');
}

//removes the focus from the input box
function td_aj_search_input_remove_focus() {
    if (td_aj_search_results !== 0) {
        jQuery('.td-search-form').css('opacity', 0.5);
    }
}

//makes an ajax request
function td_ajax_search_do_request() {

    if (jQuery('#td-header-search').val() == '') {
        td_aj_search_input_focus();
        return;
    }


    var search_query = jQuery('#td-header-search').val();


    //do we have a cache hit
    if (td_local_cache.exist(search_query)) {
        td_ajax_search_process_request(td_local_cache.get(search_query));
        return; //cache HIT
    }


    //fk no cache hit - do the real request



    jQuery.ajax({
        type: 'POST',
        url: td_ajax_url,
        data: {
            action: 'td_ajax_search',
            td_string: search_query
        },
        success: function(data, textStatus, XMLHttpRequest) {
            td_local_cache.set(search_query, data);
            td_ajax_search_process_request(data);
        },
        error: function(MLHttpRequest, textStatus, errorThrown) {
            //console.log(errorThrown);
        }
    });
}

function td_ajax_search_process_request(data) {
    var current_query = jQuery('#td-header-search').val();

    //the search is empty - drop results
    if (current_query == '') {
        jQuery('#td-aj-search').empty();
        return;
    }

    var td_data_object = jQuery.parseJSON(data); //get the data object
    //drop the result - it's from a old query
    if (td_data_object.td_search_query !== current_query) {
        return;
    }

    //reset the current selection and total posts
    td_aj_search_cur_sel = 0;
    td_aj_search_results = td_data_object.td_total_in_list;
    td_aj_first_down_up = true;


    //update the query
    jQuery('#td-aj-search').html(td_data_object.td_data);

    /*
         td_data_object.td_data
         td_data_object.td_total_results
         td_data_object.td_total_in_list
         */
}






function td_mobile_menu2() {
    //return;
    //alert('mobile menu');
    //******************************************************************
    //MOBILE MENU new

    // helper functions

    var trim = function(str)
    {
        return str.trim ? str.trim() : str.replace(/^\s+|\s+$/g, '');
    };

    var hasClass = function(el, cn)
    {
        return (' ' + el.className + ' ').indexOf(' ' + cn + ' ') !== - 1;
    };

    var addClass = function(el, cn)
    {
        if (!hasClass(el, cn)) {
            el.className = (el.className === '') ? cn : el.className + ' ' + cn;
        }
    };

    var removeClass = function(el, cn)
    {
        el.className = trim((' ' + el.className + ' ').replace(' ' + cn + ' ', ' '));
    };

    var hasParent = function(el, id)
    {
        if (el) {
            do {
                if (el.id === id) {
                    return true;
                }
                if (el.nodeType === 9) {
                    break;
                }
            }
            while ((el = el.parentNode));
        }
        return false;
    };

    // normalize vendor prefixes

    var doc = document.documentElement;

    var transform_prop = window.Modernizr.prefixed('transform'),
    transition_prop = window.Modernizr.prefixed('transition'),
    transition_end = (function() {
        var props = {
            'WebkitTransition' : 'webkitTransitionEnd',
            'MozTransition' : 'transitionend',
            'OTransition' : 'oTransitionEnd otransitionend',
            'msTransition' : 'MSTransitionEnd',
            'transition' : 'transitionend'
        };
        return props.hasOwnProperty(transition_prop) ? props[transition_prop] : false;
    })();

    window.App = (function()
    {

        var _init = false;
        var app = {};

        var inner_nav_td = document.getElementById('inner-wrap');
        var nav_open = false;
        var nav_class = 'js-nav';

        //console.log(inner_nav_td);
        app.init = function()
        {
            if (_init) {
                return;
            }
            _init = true;

            var closeNavEnd = function(e)
            {
                if (e && e.target === inner_nav_td) {
                    document.removeEventListener(transition_end, closeNavEnd, false);
                }
                nav_open = false;
            };

            app.closeNav = function()
            {
                if (nav_open) {
                    // close navigation after transition or immediately
                    //console.log(inner_nav_td);

                    jQuery('#inner-wrap').css('minHeight', 'auto');
                    var duration = (transition_end && transition_prop) ? parseFloat(window.getComputedStyle(inner_nav_td, '')[transition_prop + 'Duration']) : 0;
                    if (duration > 0) {
                        document.addEventListener(transition_end, closeNavEnd, false);
                    } else {
                        closeNavEnd(null);
                    }
                }
                removeClass(doc, nav_class);
            };

            app.openNav = function()
            {
                if (nav_open) {
                    return;
                }


                //var td_cur_innerwrap_height = jQuery('#inner-wrap').height();

                //                if (td_cur_innerwrap_height != td_cur_mobile_nav_height) {
                //
                //                }
                //                alert(td_cur_mobile_nav_height);
                var td_cur_mobile_nav_height = jQuery('.td-menu-animate').height();
                jQuery('#inner-wrap').css('minHeight', td_cur_mobile_nav_height);
                addClass(doc, nav_class);


                /*
                                jQuery(this).delay(1000).queue(function(){
                                    if (td_detect.is_phone_screen) {
                                        jQuery("html, body").animate({ scrollTop: 0 }, {
                                                duration: 1000,
                                                easing:'easeInOutQuart',
                                                complete: function(){
                                                    td_is_scrolling_animation = false;
                                                }
                                            }
                                        );
                                    }

                                    jQuery(this).dequeue();
                                });
                */


                nav_open = true;
            };

            app.toggleNav = function(e)
            {

                if (nav_open && hasClass(doc, nav_class)) {
                    app.closeNav();
                } else {
                    app.openNav();
                }
                if (e) {
                    e.preventDefault();
                }
            };

            // open nav with main "nav" button
            //document.getElementById('nav-open-btn').addEventListener('click', app.toggleNav, false);

            jQuery('#td-top-mobile-toggle a, .td-mobile-close a').click(function() {
                app.toggleNav();
            });




            // close nav with main "close" button
            //document.getElementById('nav-close-btn').addEventListener('click', app.toggleNav, false);



            addClass(doc, 'js-ready');

        };

        return app;

    })();

    if (window.addEventListener) {
        window.addEventListener('DOMContentLoaded', window.App.init, false);
    }







}




/*  ----------------------------------------------------------------------------
    Slider callbacks
 */
function slideStartedMoving(args) {
    td_is_slide_moving = true; //used on touch screens + mobile menu 2
}

function slideContentComplete(args) {
    td_is_slide_moving = false;
    if (!args.slideChanged) 
        return false;
    jQuery(args.currentSlideObject).parent().find('.slide-info-wrap').removeClass('slide-wrap-active');
    jQuery(args.currentSlideObject).children('.slide-info-wrap').addClass('slide-wrap-active');
}

function slideContentLoaded(args) {
    if (!args.slideChanged) 
        return false;
    // console.log('loaded');
    jQuery(args.currentSlideObject).parent().find('.slide-info-wrap').removeClass('slide-wrap-active');
    jQuery(args.currentSlideObject).children('.slide-info-wrap').addClass('slide-wrap-active');
}




/*  ----------------------------------------------------------------------------
    Add retina support
 */

function td_retina() {
    if (window.devicePixelRatio > 1) {
        jQuery('.td-retina').each(function(i) {
            var lowres = jQuery(this).attr('src');
            var highres = lowres.replace(".png", "@2x.png");
            highres = highres.replace(".jpg", "@2x.jpg");
            jQuery(this).attr('src', highres);

        });


        //custom logo support
        jQuery('.td-retina-data').each(function(i) {
            jQuery(this).attr('src', jQuery(this).data('retina'));
            //fix logo aligment on retina devices
            jQuery(this).addClass('td-retina-version');
        });

    }
}

/*
jQuery('body').click(function(e){
    if(! jQuery(e.target).hasClass('custom-background')){
        alert('clicked on something that has not the class theDIV');
    }

});*/

//click only on BACKGROUND, for devices that don't have touch (ex: phone, tablets)
if (!td_detect.is_touch_device && td_ad_background_click_link != '') {

    //var ev = ev || window.event;
    //var target = ev.target || ev.srcElement;
    jQuery('body').click(function(event) {

        //getting the target element that the user clicks - for W3C and MSIE
        var target = (event.target) ? event.target : event.srcElement;

        //only if the background has background image
        if (jQuery('body').hasClass('td-boxed-layout')) {
            if (target.id == 'inner-wrap' || target.className == 'td-header-menu-wrap' || target.className == 'td-header-bg' || target.className == 'td-menu-background affix-top') {

                //open the link ad page
                if (td_ad_background_click_target == '_blank') {
                    //open in a new window
                    window.open(td_ad_background_click_link)
                } else {
                    //open in the same window
                    location.href = td_ad_background_click_link;
                }
            }
        }

        //e.stopPropagation();
        //stopBubble(event);
    });
}


function stopBubble(e) {
    if (e && e.stopPropagation) {
        e.stopPropagation();
    } else {
        window.event.cancelBubble = true;
    }
}



/**
 * reading cookies
 * @param name
 * @returns {*}
 */
function td_read_site_cookie(name) {
    var nameEQ = escape(name) + "=";
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') 
            c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) == 0) 
            return unescape(c.substring(nameEQ.length, c.length));
    }
    return null;
}


/**
 *
 * @param td_time_cookie_array
 *
 * @param[0]: name of the cookie
 * @param[1]: value of the cookie
 * @param[2]: expiration time
 */
function td_set_cookies_life(td_time_cookie_array) {
    var expiry = new Date();
    expiry.setTime(expiry.getTime() + td_time_cookie_array[2]);

    // Date()'s toGMTSting() method will format the date correctly for a cookie
    document.cookie = td_time_cookie_array[0] + "=" + td_time_cookie_array[1] + "; expires=" + expiry.toGMTString() + "; path=/";
}



//get page width
function td_get_page_width() {
    var x = 0;

    if (self.innerHeight) {
        x = self.innerWidth;

    } else if (document.documentElement && document.documentElement.clientHeight) {
        x = document.documentElement.clientWidth;

    } else if (document.body) {
        x = document.body.clientWidth;
    }

    return x;
}



//function to resize the height of the slide
function td_resize_slide(args) {
    if (td_get_page_width() < 768) {
        var slide_displayd = args.currentSlideNumber;


        //console.log(args.sliderObject[0]);
        //console.log(args.data.obj[0]);

        var current_slider = jQuery(args.data.obj[0]).attr("id");

        if (td_detect.is_ie8 === false) {
            jQuery("#" + current_slider).css("overflow", "none");
            jQuery("#" + current_slider + " .item").css("overflow", "visible");
        }



        var setHeight = 0;
        setHeight = jQuery("#" + current_slider + "_item_" + slide_displayd).outerHeight(true);


        jQuery("#" + current_slider + ", #" + current_slider + " .slider").css({
            height: setHeight
        });
    }
}


//function to resize the height of the slide with jQuery.each() function
function td_resize_page_sliders() {
    jQuery(document).find('.iosSlider.td_block_big_grid').each(function() {
        var current_slider = jQuery(this).attr("id");

        if (!td_detect.is_ie8) {
            jQuery("#" + current_slider).css("overflow", "none");
            jQuery("#" + current_slider + " .item").css("overflow", "visible");
        }

        var setHeight = 0;
        setHeight = jQuery("#" + current_slider + "_item_1").outerHeight(true);

        jQuery("#" + current_slider + ", #" + current_slider + " .slider").css({
            height: setHeight
        });
    });
}

var td_loading_box = {

    //array_colors: ['#ffffff', '#fafafa', '#ececec', '#dddddd', '#bfbfbf', '#9a9a9a', '#7e7e7e', '#636363'],//whiter -> darker

    array_colors_temp: ['rgba(99, 99, 99, 0)', 'rgba(99, 99, 99, 0.05)', 'rgba(99, 99, 99, 0.08)', 'rgba(99, 99, 99, 0.2)', 'rgba(99, 99, 99, 0.3)', 'rgba(99, 99, 99, 0.5)', 'rgba(99, 99, 99, 0.6)', 'rgba(99, 99, 99, 1)'],
    //whiter -> darker

    array_colors: [],

    status_animation: 'stop',

    //stop loading box
    stop : function stop () {
        td_loading_box.status_animation = 'stop';
        //jQuery('.td-loader-gif').html("");
    },


    //init loading box
    init : function init (color) {

        var td_color_reg_exp = /^#[a-zA-Z0-9]{3,6}$/;
        if (color && td_color_reg_exp.test(color)) {

            var col_rgba = td_loading_box.hexToRgb(color);

            var rgba_string = "rgba(" + col_rgba.r + ", " + col_rgba.g + ", " + col_rgba.b + ", ";

            td_loading_box.array_colors[7] = rgba_string + " 1)";
            td_loading_box.array_colors[6] = rgba_string + " 0.6)";
            td_loading_box.array_colors[5] = rgba_string + " 0.5)";
            td_loading_box.array_colors[4] = rgba_string + " 0.3)";
            td_loading_box.array_colors[3] = rgba_string + " 0.2)";
            td_loading_box.array_colors[2] = rgba_string + " 0.08)";
            td_loading_box.array_colors[1] = rgba_string + " 0.05)";
            td_loading_box.array_colors[0] = rgba_string + " 0)";

        } else {
            //default array
            td_loading_box.array_colors = td_loading_box.array_colors_temp.slice(0);

        }

        if (td_loading_box.status_animation == 'stop') {
            td_loading_box.status_animation = 'display';
            this.render();
        }
    },


    //create the animation
    render: function render (color) {

        //call the animation_display function
        td_loading_box.animation_display('<div class="td-lb-box td-lb-box-1" style="background-color:' + td_loading_box.array_colors[0] + '"></div><div class="td-lb-box td-lb-box-2" style="background-color:' + td_loading_box.array_colors[1] + '"></div><div class="td-lb-box td-lb-box-3" style="background-color:' + td_loading_box.array_colors[2] + '"></div><div class="td-lb-box td-lb-box-4" style="background-color:' + td_loading_box.array_colors[3] + '"></div><div class="td-lb-box td-lb-box-5" style="background-color:' + td_loading_box.array_colors[4] + '"></div><div class="td-lb-box td-lb-box-6" style="background-color:' + td_loading_box.array_colors[5] + '"></div><div class="td-lb-box td-lb-box-7" style="background-color:' + td_loading_box.array_colors[6] + '"></div><div class="td-lb-box td-lb-box-8" style="background-color:' + td_loading_box.array_colors[7] + '"></div>');

        //direction right
        var temp_color_array = [td_loading_box.array_colors[0], td_loading_box.array_colors[1], td_loading_box.array_colors[2], td_loading_box.array_colors[3], td_loading_box.array_colors[4], td_loading_box.array_colors[5], td_loading_box.array_colors[6], td_loading_box.array_colors[7]];

        td_loading_box.array_colors[0] = temp_color_array[7];
        td_loading_box.array_colors[1] = temp_color_array[0];
        td_loading_box.array_colors[2] = temp_color_array[1];
        td_loading_box.array_colors[3] = temp_color_array[2];
        td_loading_box.array_colors[4] = temp_color_array[3];
        td_loading_box.array_colors[5] = temp_color_array[4];
        td_loading_box.array_colors[6] = temp_color_array[5];
        td_loading_box.array_colors[7] = temp_color_array[6];

        if (td_loading_box.status_animation == 'display') {


            setTimeout(td_loading_box.render, 40);
        } else {
            td_loading_box.animation_display('');
        }
    },


    //display the animation
    animation_display: function animation_display (animation_str) {
        jQuery('.td-loader-gif').html(animation_str);
    },


    //converts hex to rgba
    hexToRgb: function hexToRgb(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);

        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
}
//td_loading_box.init();//td_loading_box.stop();

"use strict";

/* ----------------------------------------------------------------------------
 blocks.js
 --------------------------------------------------------------------------- */



/*  ----------------------------------------------------------------------------
 On load
 */
jQuery().ready(function() {
    td_on_ready_ajax_blocks();
});





function td_on_ready_ajax_blocks() {


    /*  ----------------------------------------------------------------------------
         AJAX pagination next
         */
    jQuery(".td-ajax-next-page").click(function(event) {
        event.preventDefault();

        var current_block_obj = td_getBlockObjById(jQuery(this).data('td_block_id'));

        if (jQuery(this).hasClass('ajax-page-disabled') || current_block_obj.is_ajax_running === true) {
            return;
        }

        current_block_obj.is_ajax_running = true; // ajax is running and we're wayting for a reply from server

        current_block_obj.td_current_page++;
        td_ajax_do_block_request(current_block_obj, 'next');
    });


    /*  ----------------------------------------------------------------------------
         AJAX pagination prev
         */
    jQuery(".td_ajax-prev-page").click(function(event) {
        event.preventDefault();

        var current_block_obj = td_getBlockObjById(jQuery(this).data('td_block_id'));

        if (jQuery(this).hasClass('ajax-page-disabled') || current_block_obj.is_ajax_running === true) {
            return;
        }

        current_block_obj.is_ajax_running = true; // ajax is running and we're wayting for a reply from server

        current_block_obj.td_current_page--;
        td_ajax_do_block_request(current_block_obj, 'back');
    });


    /*  ----------------------------------------------------------------------------
         AJAX pagination load more
         */
    jQuery(".td_ajax_load_more").click(function(event) {
        event.preventDefault();
        if (jQuery(this).hasClass('ajax-page-disabled')) {
            return;
        }

        var current_block_obj = td_getBlockObjById(jQuery(this).data('td_block_id'));

        current_block_obj.td_current_page++;
        td_ajax_do_block_request(current_block_obj, 'load_more');
    });


    /*  ----------------------------------------------------------------------------
         AJAX pagination infinite load
         */
    jQuery('.td_ajax_infinite').waypoint(function(direction) {
        if (direction === 'down') {
            //console.log('loading');
            var current_block_obj = td_getBlockObjById(jQuery(this).data('td_block_id'));

            current_block_obj.td_current_page++;
            td_ajax_do_block_request(current_block_obj, 'infinite_load');
        }

    }, {
        offset: '110%' 
    });





    /*  ----------------------------------------------------------------------------
         AJAX sub cat filter
         */
    jQuery(".ajax-sub-cat").click(function(event) {
        //click on an ajax category filter
        event.preventDefault();


        //get the current block id
        var current_block_id = jQuery(this).data('td_block_id');


        //destroy any iossliders to avoid bugs
        jQuery('#' + current_block_id).find('.iosSlider').iosSlider('destroy');;

        //get current block
        var current_block_obj = td_getBlockObjById(current_block_id);

        //change cur cat
        current_block_obj.td_cur_cat = jQuery(this).data('cat_id');

        current_block_obj.td_current_page = 1;


        //console.log(current_block_obj);

        //do request
        td_ajax_do_block_request(current_block_obj, 'pull_down');
    });




    /*  ----------------------------------------------------------------------------
         AJAX MEGA MENU
         */



    var td_mega_menu_last_link = '';

    //it used to hook the mega menu categories
    function td_ajax_sub_cat_mega_run(event) {

        if (td_mega_menu_last_link != jQuery(this).attr('href')) {
            event.preventDefault();
            td_mega_menu_last_link = jQuery(this).attr('href');
        } else {
            td_mega_menu_last_link = '';
        }


        //get the current block id
        var current_block_id = jQuery(this).data('td_block_id');

        //get current block
        var current_block_obj = td_getBlockObjById(current_block_id);


        //change cur cat
        current_block_obj.td_cur_cat = jQuery(this).data('cat_id');

        current_block_obj.td_current_page = 1;

        //do request - no append, no animation
        td_ajax_do_block_request(current_block_obj, 'pull_down');
    }

    //on touch devices use click
    if (td_detect.is_touch_device) {
        jQuery(".ajax-sub-cat-mega").click(td_ajax_sub_cat_mega_run);
    } else {
        jQuery(".ajax-sub-cat-mega").hover(td_ajax_sub_cat_mega_run, function (event) {});
    }


}
// end td_on_ready_ajax_blocks()



/**
 * makes a ajax block request
 * @param current_block_obj
 * @param td_append - append the return content?
 * @returns {string}
 */
function td_ajax_do_block_request(current_block_obj, td_user_action) {

    //console.log(current_block_obj);

    //search the cache
    var current_block_obj_signature = JSON.stringify(current_block_obj);
    if (td_local_cache.exist(current_block_obj_signature)) {
        //do the animation with cache hit = true
        td_block_ajax_loading_start(current_block_obj, true);
        td_ajax_block_process_response(td_local_cache.get(current_block_obj_signature), td_user_action);
        return 'cache_hit'; //cache HIT
    }


    //cache miss - we make a full request! - cache hit - false
    td_block_ajax_loading_start(current_block_obj, false);


    jQuery.ajax({
        type: 'POST',
        url: td_ajax_url,
        cache: true,

        data: {
            action: 'td_ajax_block',
            td_atts: current_block_obj.atts,
            td_cur_cat: current_block_obj.td_cur_cat,
            td_block_id: current_block_obj.id,
            td_column_number: current_block_obj.td_column_number,
            td_current_page: current_block_obj.td_current_page,
            block_type: current_block_obj.block_type,
            td_user_action: current_block_obj.td_user_action
        },
        success: function(data, textStatus, XMLHttpRequest) {
            td_local_cache.set(current_block_obj_signature, data);
            td_ajax_block_process_response(data, td_user_action);
        },
        error: function(MLHttpRequest, textStatus, errorThrown) {
            //console.log(errorThrown);
        }
    });
}


/**
 * process the response from the ajax query (it also processes the responses stored in the cache)
 * @param data
 * @param td_user_action
 */
function td_ajax_block_process_response(data, td_user_action) {

    //read the server response
    var td_reply_obj = jQuery.parseJSON(data); //get the data object


    //console.log(td_reply_obj);
    /*
         td_data_object.td_block_id
         td_data_object.td_data
         td_data_object.td_cur_cat
         */

    jQuery('.sub-cat-' + td_reply_obj.td_block_id).removeClass('cur-sub-cat');
    jQuery('#sub-cat-' + td_reply_obj.td_block_id + '-' + td_reply_obj.td_cur_cat).addClass('cur-sub-cat');


    //load the content (in place or append)
    if (td_user_action == 'load_more' || td_user_action == 'infinite_load') {
        jQuery('#' + td_reply_obj.td_block_id).append(td_reply_obj.td_data); //append
    } else {
        jQuery('#' + td_reply_obj.td_block_id).html(td_reply_obj.td_data); //in place
    }


    //hide or show prev
    if (td_reply_obj.td_hide_prev === true) {
        jQuery('#prev-page-' + td_reply_obj.td_block_id).addClass('ajax-page-disabled');
    } else {
        jQuery('#prev-page-' + td_reply_obj.td_block_id).removeClass('ajax-page-disabled');
    }

    //hide or show next
    if (td_reply_obj.td_hide_next === true) {
        jQuery('#next-page-' + td_reply_obj.td_block_id).addClass('ajax-page-disabled');
    } else {
        jQuery('#next-page-' + td_reply_obj.td_block_id).removeClass('ajax-page-disabled');
    }


    var current_block_obj = td_getBlockObjById(td_reply_obj.td_block_id);
    if (current_block_obj.block_type === 'slide') {
        //make the first slide active (to have caption)
        jQuery('#' + td_reply_obj.td_block_id + ' .slide-wrap-active-first').addClass('slide-wrap-active');
    }

    current_block_obj.is_ajax_running = false; // finish the loading for this block


    //loading effects
    td_block_ajax_loading_end(current_block_obj, td_user_action);
}


/**
 * loading start
 * @param current_block_obj
 * @param cache_hit
 */
function td_block_ajax_loading_start(current_block_obj, cache_hit) {

    //get the element
    var el_cur_td_block_inner = jQuery('#' + current_block_obj.id);

    //remove the loader
    jQuery('.td-loader-gif').remove(); //remove any remaining loaders

    //remove animation classes
    el_cur_td_block_inner.removeClass('fadeInRight fadeInLeft fadeInDown fadeInUp animated fadeIn');

    //add overflow + fixed height
    el_cur_td_block_inner.addClass('td_block_inner_overflow');
    var td_tmp_block_height = el_cur_td_block_inner.height();
    el_cur_td_block_inner.css('height', td_tmp_block_height);


    //show the loader only if it's needed
    if (cache_hit === true) {
        el_cur_td_block_inner.stop();
    } else {
        //el_cur_td_block_inner.parent().append('<img class="td-loader-gif td-loader-animation-start" src="' + td_get_template_directory_uri + '/images/AjaxLoader.gif" alt=""/>');
        el_cur_td_block_inner.parent().append('<div class="td-loader-gif td-loader-animation-start"></div>');

        //init the loading box
        td_loading_box.init(current_block_obj.header_color);

        setTimeout(function() {
            jQuery('.td-loader-gif').removeClass('td-loader-animation-start');
            jQuery('.td-loader-gif').addClass('td-loader-animation-mid');
        }, 50);

        el_cur_td_block_inner.stop();
        el_cur_td_block_inner.fadeTo('500', 0.1, 'easeInOutCubic');
    }



    //auto height => fixed height
    var td_tmp_block_height = el_cur_td_block_inner.height();
    el_cur_td_block_inner.css('height', td_tmp_block_height);



}


/**
 * we have a reply from the ajax request
 * @param current_block_obj
 * @param td_user_action
 */
function td_block_ajax_loading_end(current_block_obj, td_user_action) {

    // remove the loader
    jQuery('.td-loader-gif').removeClass('td-loader-animation-mid');
    jQuery('.td-loader-gif').addClass('td-loader-animation-end');
    setTimeout(function() {
        jQuery('.td-loader-gif').remove();
        //stop the loading box
        td_loading_box.stop();
    }, 400);




    //get the current inner
    var el_cur_td_block_inner = jQuery('#' + current_block_obj.id);


    el_cur_td_block_inner.stop();



    switch (td_user_action) {
    case 'next':
        el_cur_td_block_inner.addClass('animated fadeInRight');
        break;
    case 'back':
        el_cur_td_block_inner.addClass('animated fadeInLeft');
        break;

    case 'pull_down':
        el_cur_td_block_inner.addClass('animated fadeInDown');
        break;

    case 'mega_menu':
        el_cur_td_block_inner.addClass('animated fadeInUp');
        break;


    case 'load_more':
        el_cur_td_block_inner.addClass('animated fadeIn');
        break;


    case 'infinite_load':
        el_cur_td_block_inner.addClass('animated fadeIn');
        break;
    }

    if (td_detect.is_ie9 || td_detect.is_ie8) {

        el_cur_td_block_inner.css('opacity', 1);
    }

    setTimeout(function() {
        jQuery('.td_block_inner_overflow').removeClass('td_block_inner_overflow');
        el_cur_td_block_inner.css('height', 'auto');

    }, 300);


    //refresh waypoints for infinit scroll
    jQuery.waypoints('refresh');
}


/**
 * search by block _id
 * @param myID - block id
 * @returns {number} the index
 */
function td_getBlockIndex(myID) {
    var cnt = 0;
    var tmpReturn = 0;
    jQuery.each(td_blocks, function(index, td_block) {
        if (td_block.id === myID) {
            tmpReturn = cnt;
            return false; //brake jquery each
        } else {
            cnt++;
        }
    });
    return tmpReturn;
}

/**
 * gets the block object using a block ID
 * @param myID
 * @returns {*} block object
 */
function td_getBlockObjById(myID) {
    return td_blocks[td_getBlockIndex(myID)];
}



/*
* Sprite3D.js - v2.0.2
* https://github.com/boblemarin/Sprite3D.js
*
* Copyright (c) 2010 boblemarin emeric@minimal.be http://www.minimal.be
* 
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
* 
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/


/*
    Modified by tagDiv 2014 -  modifications are Copyright tagDiv
 */


"use strict";

var Sprite3D = Sprite3D || {


    /********* [PUBLIC STATIC] isSupported() ***********/
    /*
    	returns: Boolean
    	This method is automatically called when we create the first element, 
    	but you can call it earlier if you want to provide an alternative content 
    	to unsupported browsers.
    	*/
    isSupported: function() {
        // init if needed
        if ( !this._isInit ) 
            this._init();
        // return support value
        return this._isSupported;
    },

    /********* [PUBLIC STATIC] stage() ***********/
    /*
    	Creates a root container for your 3D content.

    	Usage 1 : 
    		Sprite3D.stage()

    	Creates and returns a new <div> element that is added to the page.
    	The stage is centered, so the position (0,0,0) is in the center of the window.
    	This is the easiest and most common way to start a project

    	Usage 2 : 
    		Sprite3D.stage( document.querySelector("#myContainer") )

    	Uses an existing HTML element as root container. The element is only tweaked a bit,
    	adjusting a few transform-related CSS properties, as well as setting the CSS "position"
    	property to "relative" if it is "static".
    	This method gives you more freedom, but more responsabilities :)
    	
    	*/
    stage: function(element) {
        // init if needed
        if ( !this._isInit ) 
            this._init();
        // tweak or create root element
        var c, s;
        if (element) {
            c = element;
            s = element.style;
            if (s.position === "static" ) 
                s.position = "relative";
        } else {
            c = document.createElement("div");
            s = c.style;
            s[this._browserPrefix + "PerspectiveOrigin"] = "0 0";
            s[this._browserPrefix + "TransformOrigin"] = "0 0";
            s.position = "absolute";
            s.top = "50%";
            s.left = "50%";
            s.margin = "0px";
            s.padding = "0px";
            document.body.appendChild(c);
        }
        s[this._browserPrefix + "Perspective"] = "800px";
        s[this._browserPrefix + "Transform"] = "translateZ(0px)";
        c = this.create(c);
        // fix for the glitch problems under Safari6 / Mountain Lion
        // (root container must NOT have its transform-style property set to "preserve-3d")
        s[this._browserPrefix + "TransformStyle"] = "flat";
        // end fix
        return c;
    },

    /********* [PUBLIC STATIC] create() ***********/
    /*
    	Creates a new Sprite3D element

    	Usage 1 :
    		Sprite3D.create()

    		Creates a <div> element and turn it into a 

    	Usage 2 :
    		Sprite3D.create( document.querySelector("#myElement") )

    	Usage 3 :
    		Sprite3D.create( "#id" )
    		Sprite3D.create( "id" )

    	Usage 4 : 
    		Sprite3D.create( ".class" )
    		Sprite3D.create( ".class1 class2" )

    	*/
    create: function(element) {
        // init Sprite3D if needed
        if ( !this._isInit ) 
            this._init();

        // create or tweak html element
        if ( arguments.length === 0 ) {
            element = document.createElement("div");
            element.style.margin = "0px";
            element.style.padding = "0px";
            element.style.position = "absolute";
        } else if ( typeof(element) === "string" ) {
            var str = element;
            element = document.createElement("div");
            element.style.margin = "0px";
            element.style.padding = "0px";
            element.style.position = "absolute";
            this._handleStringArgument(element, str);
        } else if ( element.style.position == "static" ) {
            element.style.position = "relative";
        }
        element.style[ this._browserPrefix + "TransformStyle" ] = "preserve-3d";
        element.style[ this._transformProperty ] = "translateZ(0px)";

        // extend element with 3D methods
        for (var prop in this._props) {
            if (this._props.hasOwnProperty(prop)) {
                element[prop] = this._props[prop];
            }
        }

        // add private properties
        element._string = [
        "translate3d(", 0, "px,", 0, "px,", 0, "px) ", 
        "rotateX(", 0, "deg) ", 
        "rotateY(", 0, "deg) ", 
        "rotateZ(", 0, "deg) ", 
        "scale3d(", 1, ", ", 1, ", ", 1, ") "
        ];
        element._positions = [
        1, 3, 5, // x, y, z
        8, 11, 14, // rotationX, rotationY, rotationZ
        17, 19, 21 // scaleX, scaleY, scaleZ
        ];
        element._ox = 0;
        element._oy = 0;
        element._oz = 0;

        // return
        return element;
    },

    /********* [PUBLIC STATIC] box() ***********/
    box: function(width, height, depth, idOrClassName) {
        // init if needed
        if ( !this._isInit ) 
            this._init();

        // create container element
        var box = this.create();

        if ( arguments.length === 1 ) {
            height = width;
            depth = width;
        } else if ( arguments.length === 2 && typeof(arguments[1]) === "string" ) {
            this._handleStringArgument(box, arguments[1]);
            height = width;
            depth = width;
        } else if ( idOrClassName && typeof(idOrClassName) === "string" ) {
            this._handleStringArgument(box, idOrClassName);
        }

        // add faces
        var hwidth = width * .5,
        hheight = height * .5,
        hdepth = depth * .5;

        box.appendChild( Sprite3D.create(".front").position( - hwidth, - hheight, hdepth).size(width, height).update() );
        box.appendChild( Sprite3D.create(".back").position( - hwidth, - hheight, - hdepth).size(width, height).rotationY(180).update() );
        box.appendChild( Sprite3D.create(".left").position( - hwidth - hdepth, - hheight, 0).size(depth, height).rotationY( - 90).update() );
        box.appendChild( Sprite3D.create(".right").position( hwidth - hdepth, - hheight, 0).size(depth, height).rotationY(90).update() );
        box.appendChild( Sprite3D.create(".bottom").position( - hwidth, hheight - hdepth, 0).size(width, depth).rotationX( - 90).update() );
        box.appendChild( Sprite3D.create(".top").position( - hwidth, - hheight - hdepth, 0).size(width, depth).rotationX(90).update() );

        return box;
    },

    /********* [PUBLIC STATIC] prefix() ***********/
    prefix: function(cssPropertyName) {
        return Sprite3D._browserPrefix + cssPropertyName;
    },

    /********* [PRIVATE STATIC] library's global properties ***********/
    _isInit: false,
    _isSupported: false,
    _browserPrefix: "webkit",
    _transformProperty: "webkitTransform",

    /********* [PRIVATE STATIC] _init() ***********/
    _init: function() {
        var d = document.createElement("div"), 
        prefixes = ["", "webkit", "Moz", "O", "ms" ],
        n = prefixes.length, i;

        Sprite3D._isInit = true;
        // check for 3D transforms
        for ( i = 0; i < n; i++ ) {
            if ( ( prefixes[i] + "Perspective" ) in d.style ) {
                Sprite3D._transformProperty = prefixes[i] + "Transform";
                Sprite3D._isSupported = true;
                Sprite3D._browserPrefix = prefixes[i];
                if ( i == 2 ) 
                    Sprite3D._props.update = Sprite3D._props.updateJoin;
                //console.log( "Sprite3D found support for 3D transforms using prefix: " + prefixes[i] );
                return true;
            }
        }

        // no transform support
        alert("Sorry, but your browser does not support CSS 3D transfroms.");
        return false;
    },

    /********* [PRIVATE STATIC] _handleStringArgument() ***********/
    _handleStringArgument: function( element, str ) {
        switch ( str[0] ) {
        case ".":
            element.className = str.substr(1);
            break;
        case "#":
            element.id = str.substr(1);
            break;
        default:
            element.id = str;
            break;
        }
    },

    /********* Sprite3D objects properties ***********/
    _props: {

        /////////////////////////////////////////////
        //////////// Position / absolute ////////////
        /////////////////////////////////////////////
        x : function(px) {
            if ( arguments.length ) {
                this._string[this._positions[0]] = px - this._ox;
                return this;
            } else {
                return this._string[this._positions[0]] + this._ox;
            }
        },
        y : function(py) {
            if ( arguments.length ) {
                this._string[this._positions[1]] = py - this._oy;
                return this;
            } else {
                return this._string[this._positions[1]] + this._oy;
            }
        },
        z : function(pz) {
            if ( arguments.length ) {
                this._string[this._positions[2]] = pz - this._oz;
                return this;
            } else {
                return this._string[this._positions[2]] + this._oz;
            }
        },
        position : function( px, py, pz) {
            this._string[this._positions[0]] = px - this._ox;
            this._string[this._positions[1]] = py - this._oy;
            if ( arguments.length >= 3 ) 
                this._string[this._positions[2]] = pz - this._oz;
            return this;
        },

        /////////////////////////////////////////////
        //////////// Position / relative ////////////
        /////////////////////////////////////////////
        move : function(px, py, pz) {
            this._string[this._positions[0]] += px;
            this._string[this._positions[1]] += py;
            if ( arguments.length >= 3 ) 
                this._string[this._positions[2]] += pz;
            return this;
        },

        /////////////////////////////////////////////
        //////////// Rotation / absolute ////////////
        /////////////////////////////////////////////
        rotationX : function(rx) {
            if ( arguments.length ) {
                this._string[this._positions[3]] = rx;
                return this;
            } else {
                return this._string[this._positions[3]];
            }
        },
        rotationY : function(ry) {
            if ( arguments.length ) {
                this._string[this._positions[4]] = ry;
                return this;
            } else {
                return this._string[this._positions[4]];
            }
        },
        rotationZ : function(rz) {
            if ( arguments.length ) {
                this._string[this._positions[5]] = rz;
                return this;
            } else {
                return this._string[this._positions[5]];
            }
        },
        rotation : function(rx, ry, rz) {
            this._string[this._positions[3]] = rx;
            this._string[this._positions[4]] = ry;
            this._string[this._positions[5]] = rz;
            return this;
        },

        /////////////////////////////////////////////
        //////////// Rotation / relative ////////////
        /////////////////////////////////////////////
        rotate : function(rx, ry, rz) {
            this._string[this._positions[3]] += rx;
            this._string[this._positions[4]] += ry;
            this._string[this._positions[5]] += rz;
            return this;
        },

        /////////////////////////////////////////////
        /////////////////   Scale  //////////////////
        /////////////////////////////////////////////
        scaleX : function(sx) {
            if ( arguments.length ) {
                this._string[this._positions[6]] = sx;
                return this;
            } else {
                return this._string[this._positions[6]];
            }
        },
        scaleY : function(sy) {
            if ( arguments.length ) {
                this._string[this._positions[7]] = sy;
                return this;
            } else {
                return this._string[this._positions[7]];
            }
        },
        scaleZ : function(sz) {
            if ( arguments.length ) {
                this._string[this._positions[8]] = sz;
                return this;
            } else {
                return this._string[this._positions[8]];
            }
        },
        scale : function(sx, sy, sz) {
            switch (arguments.length) {
            case 0:
                return this._string[this._positions[6]];
            case 1:
                this._string[this._positions[6]] = sx;
                this._string[this._positions[7]] = sx;
                this._string[this._positions[8]] = sx;
                return this;
            case 2:
                this._string[this._positions[6]] = sx;
                this._string[this._positions[7]] = sy;
                //this._string[this._positions[8]] = 1;
                return this;
            case 3:
                this._string[this._positions[6]] = sx;
                this._string[this._positions[7]] = sy;
                this._string[this._positions[8]] = sz;
                return this;
            }
            return this;
        },

        /////////////////////////////////////////////
        /////////////////  Origin  //////////////////
        /////////////////////////////////////////////
        origin : function(ox, oy, oz) {
            // failed attempt at auto-centering the registration point of the object
            if ( typeof(ox) === "string" ) {
                /*
                				switch(ox){
                					case "center":
                						this._string[this._positions[0]] = -this.offsetWidth>>1;
                						this._string[this._positions[1]] = -this.offsetHeight>>1;
                						debugger
                						console.log("centering");
                						break;
                				}
                				*/
                var cs = window.getComputedStyle(this, null);
                console.log(cs);
                console.log("w:" + cs.getPropertyValue("width") + " || h: " + cs.height );
            } else {
                if (arguments.length < 3) 
                    oz = 0;
                this._string[this._positions[0]] += this._ox - ox;
                this._string[this._positions[1]] += this._oy - oy;
                this._string[this._positions[2]] += this._oz - oz;
                this._ox = ox;
                this._oy = oy;
                this._oz = oz;
            }
            return this;
        },

        /////////////////////////////////////////////
        ////////////  Transform Origin  /////////////
        /////////////////////////////////////////////
        transformOrigin : function(tx, ty) {
            this.style[ Sprite3D._browserPrefix + "TransformOrigin" ] = (Number(tx) ? tx + "px" : tx) + " " + (Number(ty) ? ty + "px" : ty);
            return this;
        },

        /////////////////////////////////////////////
        ////////////  Transform String  /////////////
        /////////////////////////////////////////////
        transformString : function(s) {
            var parts = s.toLowerCase().split(" "),
            numParts = parts.length,
            i = 0,
            strings = [],
            positions = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            n = 0;

            for (i; i < numParts; i++) {
                switch ( parts[i] ) {
                case "p":
                case "position":
                case "translate":
                    // todo: use rx ry rz (regPoint) when re-defining transform order
                    n = strings.push( "translate3d(", this._string[this._positions[0]], "px,", this._string[this._positions[1]], "px,", this._string[this._positions[2]], "px) " );
                    positions[0] = n - 6;
                    positions[1] = n - 4;
                    positions[2] = n - 2;
                    break;
                case "rx":
                case "rotatex":
                case "rotationx":
                    n = strings.push( "rotateX(", this._string[this._positions[3]], "deg) " );
                    positions[3] = n - 2;
                    break;
                case "ry" :
                case "rotatey":
                case "rotationy":
                    n = strings.push( "rotateY(", this._string[this._positions[4]], "deg) " );
                    positions[4] = n - 2;
                    break;
                case "rz":
                case "rotatez":
                case "rotationz":
                    n = strings.push( "rotateZ(", this._string[this._positions[5]], "deg) " );
                    positions[5] = n - 2;
                    break;
                case "s":
                case "scale":
                    n = strings.push( "scale3d(", this._string[this._positions[6]], ",", this._string[this._positions[7]], ",", this._string[this._positions[8]], ") " );
                    positions[6] = n - 6;
                    positions[7] = n - 4;
                    positions[8] = n - 2;
                    break;
                }
            }

            this._string = strings;
            this._positions = positions;

            return this;
        },

        /////////////////////////////////////////////
        /////////////////  Update  //////////////////
        /////////////////////////////////////////////	
        updateJoin : function() {
            this.style[Sprite3D._transformProperty] = this._string.join("");
            return this;
        },

        update : function() {
            var s = "";
            this._string.every( function(value) {
                s += value;
                return true;
            });
            this.style[Sprite3D._transformProperty] = s;
            return this;
        },

        /////////////////////////////////////////////
        ////////////  Helper Functions //////////////
        /////////////////////////////////////////////

        //////////// Perspective helper function ////////////
        perspective : function(value) {
            switch (arguments.length) {
            case 0:
                return this.style[Sprite3D._browserPrefix + "Perspective"];

            case 1:
                this.style[Sprite3D._browserPrefix + "Perspective"] = (typeof(value) === "string") ? value : value + "px";
                return this;
            }
        },

        //////////// CSS helper function ////////////
        css : function(name, value) {
            switch (arguments.length) {
            case 0:
                return this.style;
            case 1:
                return this.style[name];
            case 2:
                this.style[name] = value;
                return this;
            case 3:
                this.style[Sprite3D._browserPrefix + name] = value;
                return this;
            }
        },

        //////////// Class names helper functions ////////////
        addClass : function(name) {
            this.classList.add(name);
            return this;
        },

        removeClass : function(name) {
            this.classList.remove(name);
            return this;
        },

        //////////// HTML helper function ////////////
        html : function(value) {
            if (arguments.length) {
                this.innerHTML = value;
                return this;
            } else {
                return this.innerHTML;
            }
            return this;
        },

        //////////// SIZE helper function ////////////
        size: function(width, height) {
            this.style.width = Number(width) ? width + "px" : width;
            this.style.height = Number(height) ? height + "px" : height;
            return this;
        },

        //////////// BIND helper function ////////////
        bind: function(events) {
            if (typeof(events) === "object") {
                for ( var i in events ) {
                    this.addEventListener( i, events[i], false );
                }
            } else if (arguments.length === 2) {
                this.addEventListener( arguments[0], arguments[1], false );
            } else if (arguments.length === 3) {
                this.addEventListener( arguments[0], arguments[1], arguments[2] );
            }
            return this;
        },

        //////////// UNBIND helper function ////////////
        unbind: function(events) {
            if (typeof(events) === "object") {
                for ( var i in events ) {
                    this.removeEventListener( i, events[i], false );
                }
            } else if (arguments.length === 2) {
                this.removeEventListener( arguments[0], arguments[1], false );
            }
            return this;
        },

        //////////// Spritesheet helper functions ////////////
        tileWidth: 0,
        tileHeight: 0,
        tileSize : function(width, height) {
            this.tileWidth = width;
            this.tileHeight = height;
            return this;
        },
        tilePosition : function(tilePosX, tilePosY) {
            this.style.backgroundPosition = "-" + (tilePosX * this.tileWidth) + "px -" + (tilePosY * this.tileHeight) + "px";
            return this;
        },

        //////////// Generic setter for chaining ////////////
        set : function(name, value) {
            this[name] = value;
            return this;
        }

    }
};

/**
 * td_events.js - handles the events that requiere throttling
 * Created by ra on 6/27/14.
 * v1.1
 */

var td_events = {

    //the events - we have timers that look at the variables and fire the event if the flag is true
    scroll_event_slow_run: false,
    resize_event_run: false,
    //when true, fire up the resize event



    scroll_window_scrollTop: 0,
    //used to store the scrollTop

    init: function init() {

        jQuery(window).scroll(function() {
            td_events.scroll_event_slow_run = true;


            //read the scroll top
            td_events.scroll_window_scrollTop = jQuery(window).scrollTop();

            /*  ----------------------------------------------------------------------------
                            Run affix menu event
                         */
            td_affix.td_events_scroll(td_events.scroll_window_scrollTop); //main menu
        });


        jQuery(window).resize(function() {
            td_events.resize_event_run = true;
        });



        //low resolution timer for rest?
        setInterval(function() {
            //scroll event
            if (td_events.scroll_event_slow_run) {
                td_events.scroll_event_slow_run = false;

                //back to top
                td_events_scroll_scroll_to_top(td_events.scroll_window_scrollTop);

                //more articles box
                td_more_articles_box.td_events_scroll(td_events.scroll_window_scrollTop);

            }

            //resize event
            if (td_events.resize_event_run) {
                td_events.resize_event_run = false;
                td_affix.compute_top();
                td_detect.run_is_phone_screen();
            }
        }, 500);

    }



}

td_events.init();

"use strict";

/* ----------------------------------------------------------------------------
 td_post_images.js
 --------------------------------------------------------------------------- */



/*  ----------------------------------------------------------------------------
 On load
 */
jQuery().ready(function() {
    //handles the modal images
    td_modal_image();

    //move classes from post images to figure - td-post-image-full etc
    //td_util.image_move_class_to_figure('td-post-image-full');
    //td_util.image_move_class_to_figure('td-post-image-right');
    //td_util.image_move_class_to_figure('td-post-image-left');




});









var td_modal_image_last_el = '';

// handles modal images for: Featured images, inline image, inline image with caption, galleries
function td_modal_image() {

    //fix wordpress figure + figcaption (we move the figcaption in the data-caption attribute of the link)
    jQuery('figure.wp-caption').each(function() {
        var caption_text = jQuery(this).children('figcaption').html();
        jQuery(this).children('a').data('caption', caption_text);
    });

    //move td-modal-image class to the parent a from the image. We can only add this class to the image via word press media editor
    jQuery('.td-modal-image').each(function() {
        jQuery(this).parent().addClass('td-modal-image');
        jQuery(this).removeClass('td-modal-image');
    });



    //popup on modal images in articles
    jQuery('article').magnificPopup({
        type: 'image',
        delegate: ".td-modal-image",
        gallery: {
            enabled: true
        },
        image: {
            tError: "<a href=\'%url%\'>The image #%curr%</a> could not be loaded.",
            titleSrc: function(item) {
                //console.log(item.el);
                //alert(jQuery(item.el).data("caption"));
                var td_current_caption = jQuery(item.el).data('caption');
                if (typeof td_current_caption != "undefined") {
                    return td_current_caption;
                } else {
                    return '';
                }


            }
        },
        zoom: {
            enabled: true,
            duration: 300,
            opener: function(element) {
                return element.find("img");
            }
        },
        callbacks: {
            change: function(item) {
                td_modal_image_last_el = item.el;
                //setTimeout(function(){
                td_util.scroll_into_view(item.el);
                //}, 100);

            },
            beforeClose: function() {
                td_util.scroll_into_view(td_modal_image_last_el);
            }

        }



    });





    //gallery popup
    //detect jetpack carousel and disable the theme popup
    if (typeof jetpackCarouselStrings === 'undefined') {

        // copy gallery caption from figcaption to data-caption attribute of the link to the full image, in this way the modal can read the caption
        jQuery('figure.gallery-item').each(function() {
            var caption_text = jQuery(this).children('figcaption').html();
            jQuery(this).find('a').data('caption', caption_text);
        });



        //jquery tiled gallery
        jQuery('.tiled-gallery').magnificPopup({
            type: 'image',
            delegate: "a",
            gallery: {
                enabled: true
            },
            image: {
                tError: "<a href=\'%url%\'>The image #%curr%</a> could not be loaded.",
                titleSrc: function(item) {
                    //console.log(item.el);
                    var td_current_caption = jQuery(item.el).parent().find('.tiled-gallery-caption').text();
                    if (typeof td_current_caption != "undefined") {
                        return td_current_caption;
                    } else {
                        return '';
                    }
                }
            },
            zoom: {
                enabled: true,
                duration: 300,
                opener: function(element) {
                    return element.find("img");
                }
            },
            callbacks: {
                change: function(item) {
                    td_modal_image_last_el = item.el;
                    td_util.scroll_into_view(item.el);
                },
                beforeClose: function() {
                    td_util.scroll_into_view(td_modal_image_last_el);
                }

            }
        });



        jQuery('.gallery').magnificPopup({
            type: 'image',
            delegate: ".gallery-icon > a",
            gallery: {
                enabled: true
            },
            image: {
                tError: "<a href=\'%url%\'>The image #%curr%</a> could not be loaded.",
                titleSrc: function(item) {
                    //console.log(item.el);
                    var td_current_caption = jQuery(item.el).data('caption');
                    if (typeof td_current_caption != "undefined") {
                        return td_current_caption;
                    } else {
                        return '';
                    }
                }
            },
            zoom: {
                enabled: true,
                duration: 300,
                opener: function(element) {
                    return element.find("img");
                }
            },
            callbacks: {
                change: function(item) {
                    td_modal_image_last_el = item.el;
                    td_util.scroll_into_view(item.el);
                },
                beforeClose: function() {
                    td_util.scroll_into_view(td_modal_image_last_el);
                }

            }
        });


    }

}
//end modal

/**
 * Created by ra on 7/9/14.
 */


/*
 * tagDiv mods:
 * - added the td-backstrach class
 * - changed the backstretch.after event so that it dosn't wait for fade
 */

/*
 * Backstretch
 * http://srobbin.com/jquery-plugins/backstretch/
 *
 * Copyright (c) 2013 Scott Robbin
 * Licensed under the MIT license.
 */

;
(function ($, window, undefined) {
    'use strict';

    /* PLUGIN DEFINITION
         * ========================= */

    $.fn.backstretch = function (images, options) {
        // We need at least one image or method name
        if (images === undefined || images.length === 0) {
            $.error("No images were supplied for Backstretch");
        }

        /*
                 * Scroll the page one pixel to get the right window height on iOS
                 * Pretty harmless for everyone else
                 */
        if ($(window).scrollTop() === 0 ) {
            window.scrollTo(0, 0);
        }

        return this.each(function () {
            var $this = $(this)
            , obj = $this.data('backstretch');

            // Do we already have an instance attached to this element?
            if (obj) {

                // Is this a method they're trying to execute?
                if (typeof images == 'string' && typeof obj[images] == 'function') {
                    // Call the method
                    obj[images](options);

                    // No need to do anything further
                    return;
                }

                // Merge the old options with the new
                options = $.extend(obj.options, options);

                // Remove the old instance
                obj.destroy(true);
            }

            obj = new Backstretch(this, images, options);
            $this.data('backstretch', obj);
        });
    };

    // If no element is supplied, we'll attach to body
    $.backstretch = function (images, options) {
        // Return the instance
        return $('body')
        .backstretch(images, options)
        .data('backstretch');
    };

    // Custom selector
    $.expr[':'].backstretch = function(elem) {
        return $(elem).data('backstretch') !== undefined;
    };

    /* DEFAULTS
         * ========================= */

    $.fn.backstretch.defaults = {
        centeredX: true // Should we center the image on the X axis?
        ,
        centeredY: true // Should we center the image on the Y axis?
        ,
        duration: 5000 // Amount of time in between slides (if slideshow)
        ,
        fade: 0 // Speed of fade transition between slides
    };

    /* STYLES
         *
         * Baked-in styles that we'll apply to our elements.
         * In an effort to keep the plugin simple, these are not exposed as options.
         * That said, anyone can override these in their own stylesheet.
         * ========================= */
    var styles = {
        wrap: {
            left: 0
            ,
            top: 0
            ,
            overflow: 'hidden'
            ,
            margin: 0
            ,
            padding: 0
            ,
            height: '100%'
            ,
            width: '100%'
            ,
            zIndex: - 999999
        },
        img: {
            position: 'absolute'
            ,
            display: 'none'
            ,
            margin: 0
            ,
            padding: 0
            ,
            border: 'none'
            ,
            width: 'auto'
            ,
            height: 'auto'
            ,
            maxHeight: 'none'
            ,
            maxWidth: 'none'
            ,
            zIndex: - 999999
        }
    };

    /* CLASS DEFINITION
         * ========================= */
    var Backstretch = function (container, images, options) {
        this.options = $.extend({}, $.fn.backstretch.defaults, options || {});

        /* In its simplest form, we allow Backstretch to be called on an image path.
                 * e.g. $.backstretch('/path/to/image.jpg')
                 * So, we need to turn this back into an array.
                 */
        this.images = $.isArray(images) ? images : [images];

        // Preload images
        $.each(this.images, function () {
            $('<img />')[0].src = this;
        });

        // Convenience reference to know if the container is body.
        this.isBody = container === document.body;

        /* We're keeping track of a few different elements
                 *
                 * Container: the element that Backstretch was called on.
                 * Wrap: a DIV that we place the image into, so we can hide the overflow.
                 * Root: Convenience reference to help calculate the correct height.
                 */
        this.$container = $(container);
        this.$root = this.isBody ? supportsFixedPosition ? $(window) : $(document) : this.$container;

        // Don't create a new wrap if one already exists (from a previous instance of Backstretch)
        var $existing = this.$container.children(".backstretch").first();
        this.$wrap = $existing.length ? $existing : $('<div class="backstretch"></div>').css(styles.wrap).appendTo(this.$container);

        // Non-body elements need some style adjustments
        if (!this.isBody) {
            // If the container is statically positioned, we need to make it relative,
            // and if no zIndex is defined, we should set it to zero.
            var position = this.$container.css('position')
            , zIndex = this.$container.css('zIndex');

            this.$container.css({
                position: position === 'static' ? 'relative' : position
                ,
                zIndex: zIndex === 'auto' ? 0 : zIndex
                ,
                background: 'none'
            });

            // Needs a higher z-index
            this.$wrap.css({
                zIndex: - 999998
            });
        }

        // Fixed or absolute positioning?
        this.$wrap.css({
            position: this.isBody && supportsFixedPosition ? 'fixed' : 'absolute'
        });

        // Set the first image
        this.index = 0;
        this.show(this.index);

        // Listen for resize
        $(window).on('resize.backstretch', $.proxy(this.resize, this))
        .on('orientationchange.backstretch', $.proxy(function () {
            // Need to do this in order to get the right window height
            if (this.isBody && window.pageYOffset === 0) {
                window.scrollTo(0, 1);
                this.resize();
            }
        }, this));
    };

    /* PUBLIC METHODS
         * ========================= */
    Backstretch.prototype = {
        resize: function () {
            try {
                var bgCSS = {
                    left: 0,
                    top: 0
                }, rootWidth = this.isBody ? this.$root.width() : this.$root.innerWidth()
                , bgWidth = rootWidth
                , rootHeight = this.isBody ? ( window.innerHeight ? window.innerHeight : this.$root.height() ) : this.$root.innerHeight()
                , bgHeight = bgWidth / this.$img.data('ratio')
                , bgOffset;

                // Make adjustments based on image ratio
                if (bgHeight >= rootHeight) {
                    bgOffset = (bgHeight - rootHeight) / 2;
                    if (this.options.centeredY) {
                        bgCSS.top = '-' + bgOffset + 'px';
                    }
                } else {
                    bgHeight = rootHeight;
                    bgWidth = bgHeight * this.$img.data('ratio');
                    bgOffset = (bgWidth - rootWidth) / 2;
                    if (this.options.centeredX) {
                        bgCSS.left = '-' + bgOffset + 'px';
                    }
                }

                this.$wrap.css({
                    width: rootWidth,
                    height: rootHeight
                })
                .find('img:not(.deleteable)').css({
                    width: bgWidth,
                    height: bgHeight
                }).css(bgCSS);
            } catch (err) {
                // IE7 seems to trigger resize before the image is loaded.
                // This try/catch block is a hack to let it fail gracefully.
            }

            return this;
        }

        // Show the slide at a certain position
        ,
        show: function (newIndex) {

            // Validate index
            if (Math.abs(newIndex) > this.images.length - 1) {
                return;
            }

            // Vars
            var self = this
            , oldImage = self.$wrap.find('img').addClass('deleteable')
            , evtOptions = {
                relatedTarget: self.$container[0] 
            };

            // Trigger the "before" event
            self.$container.trigger($.Event('backstretch.before', evtOptions), [self, newIndex]);

            // Set the new index
            this.index = newIndex;

            // Pause the slideshow
            clearInterval(self.interval);

            // New image
            self.$img = $('<img />')
            .css(styles.img)
            .addClass('td-backstretch')
            .bind('load', function (e) {
                var imgWidth = this.width || $(e.target).width()
                , imgHeight = this.height || $(e.target).height();

                // Save the ratio
                $(this).data('ratio', imgWidth / imgHeight);

                // Show the image, then delete the old one
                // "speed" option has been deprecated, but we want backwards compatibilty
                $(this).fadeIn(self.options.speed || self.options.fade, function () {
                    oldImage.remove();

                    // Resume the slideshow
                    if (!self.paused) {
                        self.cycle();
                    }


                });

                // Trigger the "after" and "show" events
                // "show" is being deprecated
                $(['after', 'show']).each(function () {
                    self.$container.trigger($.Event('backstretch.' + this, evtOptions), [self, newIndex]);
                });

                // Resize
                self.resize();
            })
            .appendTo(self.$wrap);

            // Hack for IE img onload event
            self.$img.attr('src', self.images[newIndex]);
            return self;
        }
,
        next: function () {
            // Next slide
            return this.show(this.index < this.images.length - 1 ? this.index + 1 : 0);
        }
,
        prev: function () {
            // Previous slide
            return this.show(this.index === 0 ? this.images.length - 1 : this.index - 1);
        }
,
        pause: function () {
            // Pause the slideshow
            this.paused = true;
            return this;
        }
,
        resume: function () {
            // Resume the slideshow
            this.paused = false;
            this.next();
            return this;
        }
,
        cycle: function () {
            // Start/resume the slideshow
            if (this.images.length > 1) {
                // Clear the interval, just in case
                clearInterval(this.interval);

                this.interval = setInterval($.proxy(function () {
                    // Check for paused slideshow
                    if (!this.paused) {
                        this.next();
                    }
                }, this), this.options.duration);
            }
            return this;
        }
,
        destroy: function (preserveBackground) {
            // Stop the resize events
            $(window).off('resize.backstretch orientationchange.backstretch');

            // Clear the interval
            clearInterval(this.interval);

            // Remove Backstretch
            if (!preserveBackground) {
                this.$wrap.remove();
            }
            this.$container.removeData('backstretch');
        }
    };

    /* SUPPORTS FIXED POSITION?
         *
         * Based on code from jQuery Mobile 1.1.0
         * http://jquerymobile.com/
         *
         * In a nutshell, we need to figure out if fixed positioning is supported.
         * Unfortunately, this is very difficult to do on iOS, and usually involves
         * injecting content, scrolling the page, etc.. It's ugly.
         * jQuery Mobile uses this workaround. It's not ideal, but works.
         *
         * Modified to detect IE6
         * ========================= */

    var supportsFixedPosition = (function () {
        var ua = navigator.userAgent
        , platform = navigator.platform
        // Rendering engine is Webkit, and capture major version
        , wkmatch = ua.match( /AppleWebKit\/([0-9]+)/ )
        , wkversion = !!wkmatch && wkmatch[ 1 ]
        , ffmatch = ua.match( /Fennec\/([0-9]+)/ )
        , ffversion = !!ffmatch && ffmatch[ 1 ]
        , operammobilematch = ua.match( /Opera Mobi\/([0-9]+)/ )
        , omversion = !!operammobilematch && operammobilematch[ 1 ]
        , iematch = ua.match( /MSIE ([0-9]+)/ )
        , ieversion = !!iematch && iematch[ 1 ];

        return !(
        // iOS 4.3 and older : Platform is iPhone/Pad/Touch and Webkit version is less than 534 (ios5)
        ((platform.indexOf( "iPhone" ) > - 1 || platform.indexOf( "iPad" ) > - 1 || platform.indexOf( "iPod" ) > - 1 ) && wkversion && wkversion < 534) ||
        // Opera Mini
        (window.operamini && ({}).toString.call( window.operamini ) === "[object OperaMini]") ||
        (operammobilematch && omversion < 7458) ||
        //Android lte 2.1: Platform is Android and Webkit version is less than 533 (Android 2.2)
        (ua.indexOf( "Android" ) > - 1 && wkversion && wkversion < 533) ||
        // Firefox Mobile before 6.0 -
        (ffversion && ffversion < 6) ||
        // WebOS less than 3
        ("palmGetResource" in window && wkversion && wkversion < 534) ||
        // MeeGo
        (ua.indexOf( "MeeGo" ) > - 1 && ua.indexOf( "NokiaBrowser/8.5.0" ) > - 1) ||
        // IE6
        (ieversion && ieversion <= 6)
        );
    }());

}(jQuery, window));

/**
 * Created by ra on 7/8/14.
 */

jQuery().ready(function() {




    jQuery(window).on("backstretch.after", function (e, instance, index) {
        td_template_single_1();
    });
});


/**
 * make td-post-template-6 title move down and blurry
 */
function td_template_single_1() {
    //run only on a post with template 6 on it
    if (jQuery("#td-full-screen-header-image").length > 0) {


        //define all the variables - for better performance ?
        //var td_parallax_el = document.getElementById('td_parallax_header_6');

        var td_parallax_bg_el = jQuery(".td-backstretch")[0];


        //console.log(td_parallax_bg_el);
        var scroll_from_top = '';
        var distance_from_bottom;

        //attach the animation tick on scroll
        jQuery(window).scroll(function() {
            // with each scroll event request an animation frame (we have a polyfill for animation frame)
            // the requestAnimationFrame is called only once and after that we wait
            td_request_tick();
        });
    }


    var td_animation_running = false; //if the tick is running, we set this to true

    function td_request_tick() {
        if (td_animation_running === false) {
            window.requestAnimationFrame(td_do_animation);
        }
        td_animation_running = true;
    }

    /**
         * the animation loop
         */
    function td_do_animation() {
        scroll_from_top = jQuery(document).scrollTop();
        if (scroll_from_top <= 950) {
            //stop the animation after scroll from top

            var blur_value = 1 - (scroll_from_top / 800); // @todo trebuie verificata formula??
            if (td_detect.is_ie8 === true) {
                blur_value = 1;
            }


            blur_value = Math.round(blur_value * 100) / 100;

            //opacity
            //td_parallax_el.style.opacity = blur_value;

            //move the bg
            var parallax_move = - Math.round(scroll_from_top / 4);
            td_move_Y(td_parallax_bg_el, - parallax_move);


            //move the title + cat
            distance_from_bottom = - Math.round(scroll_from_top / 8);
            //td_move_Y(td_parallax_el,-distance_from_bottom);
            //td_parallax_el.style.bottom = distance_from_bottom + "px";  //un accelerated version


        }

        td_animation_running = false;
    }


    function td_move_Y (elm, value) {
        var translate = 'translate3d(0px,' + value + 'px, 0px)';
        elm.style['-webkit-transform'] = translate;
        elm.style['-moz-transform'] = translate;
        elm.style['-ms-transform'] = translate;
        elm.style['-o-transform'] = translate;
        elm.style.transform = translate;
    }

}

/*
    td_util.js
    v1.1
 */

"use strict";


/*  ----------------------------------------------------------------------------
 On load
 */
jQuery().ready(function() {

    /**
         * Modal window js code
         */
    jQuery('.td-login-modal-js').magnificPopup({
        type: 'inline',
        preloader: false,
        focus: '#name',
        removalDelay: 500,

        // When elemened is focused, some mobile browsers in some cases zoom in
        // It looks not nice, so we disable it:
        callbacks: {
            beforeOpen: function() {


                this.st.mainClass = this.st.el.attr('data-effect');


                //empty all fields
                td_modala_empty_all_fields();

                //empty error display div
                td_modala_empty_err_div();

                if (jQuery(window).width() < 700) {
                    this.st.focus = false;
                } else {
                    if (td_detect.is_ie === false) {
                        //do not focus on ie 10
                        this.st.focus = '#login_email';
                    }

                }
            },

            beforeClose: function() {}
        }
    });


    //login
    jQuery('#login-link').on( "click", function() {
        //hides or shows the divs with inputs
        show_hide_content_modala([['#td-login-div', 1], ['#td-register-div', 0], ['#td-forgot-pass-div', 0]]);

        //moves focus on the tab
        modala_swich_tabs([['#login-link', 1], ['#register-link', 0]]);

        if (jQuery(window).width() > 700 && td_detect.is_ie === false) {
            jQuery('#login_email').focus();
        }

        //empty all fields
        //td_modala_empty_all_fields();

        //empty error display div
        td_modala_empty_err_div();
    });
    //login button
    jQuery('#login_button').on( "click", function() {
        handle_login_for_modal_window();
    });
    //enter key on #login_pass
    jQuery('#login_pass').keydown(function(event) {
        if ((event.which && event.which == 13) || (event.keyCode && event.keyCode == 13)) {
            handle_login_for_modal_window();
        }
    });

    //register
    jQuery('#register-link').on( "click", function() {
        //hides or shows the divs with inputs
        show_hide_content_modala([['#td-login-div', 0], ['#td-register-div', 1], ['#td-forgot-pass-div', 0]]);

        //moves focus on the tab
        modala_swich_tabs([['#login-link', 0], ['#register-link', 1]]);

        if (jQuery(window).width() > 700 && td_detect.is_ie === false) {
            jQuery('#register_email').focus();
        }

        //empty all fields
        //td_modala_empty_all_fields();

        //empty error display div
        td_modala_empty_err_div();
    });
    //register button
    jQuery('#register_button').on( "click", function() {
        handle_register_for_modal_window();
    });
    //enter key on #register_user
    jQuery('#register_user').keydown(function(event) {
        if ((event.which && event.which == 13) || (event.keyCode && event.keyCode == 13)) {
            handle_register_for_modal_window();
        }
    });

    //forgot pass
    jQuery('#forgot-pass-link').on( "click", function() {
        //hides or shows the divs with inputs
        show_hide_content_modala([['#td-login-div', 0], ['#td-register-div', 0], ['#td-forgot-pass-div', 1]]);

        //moves focus on the tab
        modala_swich_tabs([['#login-link', 0], ['#register-link', 0]]);

        if (jQuery(window).width() > 700 && td_detect.is_ie === false) {
            jQuery('#forgot_email').focus();
        }

        //empty all fields
        //td_modala_empty_all_fields();

        //empty error display div
        td_modala_empty_err_div();
    });
    //forgot button
    jQuery('#forgot_button').on( "click", function() {
        handle_forgot_password_for_modal_window();
    });
    //enter key on #forgot_email
    jQuery('#forgot_email').keydown(function(event) {
        if ((event.which && event.which == 13) || (event.keyCode && event.keyCode == 13)) {
            handle_forgot_password_for_modal_window();
        }
    });


}); //end jquery ready



//patern to check emails
var td_mod_pattern_email = /^[a-zA-Z0-9][a-zA-Z0-9_\.-]{0,}[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9_\.-]{0,}[a-z0-9][\.][a-z0-9]{2,4}$/;

/**
 * handle all request made from login tab
 */
function handle_login_for_modal_window() {
    var login_email = jQuery('#login_email').val();
    var login_pass = jQuery('#login_pass').val();

    if (login_email && login_pass) {
        //empty error display div
        //td_modala_empty_err_div();

        modala_add_remove_class(['.td_display_err', 1, "td_display_msg_ok"]);
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_please_wait);

        //call ajax for log in
        td_modala_call_ajax('td_mod_login', login_email, '', login_pass);
    } else {
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_email_user_pass_incorrect);
    }
}

/**
 * handle all request made from register tab
 */
function handle_register_for_modal_window() {
    var register_email = jQuery('#register_email').val();
    var register_user = jQuery('#register_user').val();

    if (td_mod_pattern_email.test(register_email) && register_user) {
        //empty error display div
        //td_modala_empty_err_div();

        modala_add_remove_class(['.td_display_err', 1, "td_display_msg_ok"]);
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_please_wait);

        //call ajax
        td_modala_call_ajax('td_mod_register', register_email, register_user, '');
    } else {
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_email_user_incorrect);
    }
}

/**
 * handle all request made from forgot password tab
 */
function handle_forgot_password_for_modal_window() {
    var forgot_email = jQuery('#forgot_email').val();

    if (td_mod_pattern_email.test(forgot_email)) {
        //empty error display div
        //td_modala_empty_err_div();

        modala_add_remove_class(['.td_display_err', 1, "td_display_msg_ok"]);
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_please_wait);

        //call ajax
        td_modala_call_ajax('td_mod_remember_pass', forgot_email, '', '');
    } else {
        jQuery('.td_display_err').show();
        td_modala_write_err_div(td_email_incorrect);
    }
}

/**
 * swhich the div's acordingly to the user action (Log In, Register, Remember Password)
 *
 * ids_array : array of ids that have to be showed or hidden
 */
function show_hide_content_modala(ids_array) {
    var length = ids_array.length;

    for (var i = 0; i < length; i++) {
        var element_id = ids_array[i][0];
        var element_visibility = ids_array[i][1];

        if (element_visibility == 1) {
            jQuery(element_id).removeClass('td-dispaly-none').addClass('td-dispaly-block');
        } else {
            jQuery(element_id).removeClass('td-dispaly-block').addClass('td-dispaly-none');
        }
    }
}


/**
 * swhich the tab's acordingly to the user action (Log In, Register, Remember Password)
 *
 * ids_array : array of ids that have to be focus on or unfocus
 */
function modala_swich_tabs(ids_array) {
    var length = ids_array.length;

    for (var i = 0; i < length; i++) {
        var element_id = ids_array[i][0];
        var element_visibility = ids_array[i][1];

        if (element_visibility == 1) {
            jQuery(element_id).addClass('td_login_tab_focus');
        } else {
            jQuery(element_id).removeClass('td_login_tab_focus');
        }
    }
}


/**
 * adds or remove a class from an html object
 *
 * param : array with object identifier (id - # or class - .)
 * ex: ['.class_indetifier', 1, 'class_to_add'] or ['.class_indetifier', 0, 'class_to_remove']
 */
function modala_add_remove_class(param) {

    //add class
    if (param[1] == 1) {
        jQuery(param[0]).addClass(param[2]);

        //remove class
    } else {
        jQuery(param[0]).removeClass(param[2]);
    }
}


/**
 * empty the error div
 */
function td_modala_empty_err_div() {
    jQuery('.td_display_err').html('');
    jQuery('.td_display_err').hide();
}


/**
 * write text to error div
 */
function td_modala_write_err_div(message) {
    jQuery('.td_display_err').html(message);
}

/**
 * empty all fields in modal window
 */
function td_modala_empty_all_fields() {
    //login fields
    jQuery('#login_email').val('');
    jQuery('#login_pass').val('');

    //register fields
    jQuery('#register_email').val('');
    jQuery('#register_user').val('');

    //forgot pass
    jQuery('#forgot_email').val('');
}


/**
 * call to server from modal window
 *
 * @param $action : what action (log in, register, forgot email)
 * @param $email  : the email beening sent
 * @param $user   : the user name beening sent
 */
function td_modala_call_ajax(sent_action, sent_email, sent_user, sent_pass) {
    jQuery.ajax({
        type: 'POST',
        url: td_ajax_url,
        data: {
            action: sent_action,
            email: sent_email,
            user: sent_user,
            pass: sent_pass
        },
        success: function(data, textStatus, XMLHttpRequest) {
            var td_data_object = jQuery.parseJSON(data); //get the data object

            //check the response from server
            switch (td_data_object[0]) {
            case 'login':
                if (td_data_object[1] == 1) {
                    location.reload(true);
                } else {
                    modala_add_remove_class(['.td_display_err', 0, 'td_display_msg_ok']);
                    jQuery('.td_display_err').show();
                    td_modala_write_err_div(td_data_object[2]);
                }
                break;

            case 'register':
                if (td_data_object[1] == 1) {
                    modala_add_remove_class(['.td_display_err', 1, "td_display_msg_ok"]);
                    jQuery('.td_display_err').show();
                } else {
                    modala_add_remove_class(['.td_display_err', 0, "td_display_msg_ok"]);
                    jQuery('.td_display_err').show();
                }
                td_modala_write_err_div(td_data_object[2]);
                break;

            case 'remember_pass':
                if (td_data_object[1] == 1) {
                    modala_add_remove_class(['.td_display_err', 1, "td_display_msg_ok"]);
                    jQuery('.td_display_err').show();
                } else {
                    modala_add_remove_class(['.td_display_err', 0, "td_display_msg_ok"]);
                    jQuery('.td_display_err').show();
                }
                td_modala_write_err_div(td_data_object[2]);
                break;

            }


        },
        error: function(MLHttpRequest, textStatus, errorThrown) {
            //console.log(errorThrown);
        }
    });
}

/*  ----------------------------------------------------------------------------
 tagDiv live css compiler ( 2013 )
 - this script is used on our demo site to customize the theme live
 - not used on production sites
 */


var td_style_buffer =
'<style> /* @theme_color */ .category .entry-content, .tag .entry-content, .td_quote_box { -webkit-transition: border-color 1s ease-in-out; -moz-transition: border-color 1s ease-in-out; -o-transition: border-color 1s ease-in-out; transition: border-color 1s ease-in-out; border-color: @theme_color; } .block-title a, .block-title span, .td-tags a:hover, .td-scroll-up-visible, .td-scroll-up, .sf-menu ul .current-menu-item > a, .sf-menu ul a:hover, .sf-menu ul .sfHover > a, .td-rating-bar-wrap div, .iosSlider .slide-meta-cat, .sf-menu ul .current-menu-ancestor > a, .td-404-sub-sub-title a, .widget_tag_cloud .tagcloud a:hover, .td-mobile-close a, ul.td-category a, .td_social .td_social_type .td_social_button a, .dropcap { -webkit-transition: background-color 1s ease-in-out; -moz-transition: background-color 1s ease-in-out; -o-transition: background-color 1s ease-in-out; transition: background-color 1s ease-in-out; background-color: @theme_color; } .block-title, .sf-menu li a:hover, .sf-menu .sfHover a, .sf-menu .current-menu-ancestor a, .header-search-wrap .dropdown-menu, .sf-menu > .current-menu-item > a, .ui-tabs-nav { -webkit-transition: border-color 1s ease-in-out; -moz-transition: border-color 1s ease-in-out; -o-transition: border-color 1s ease-in-out; transition: border-color 1s ease-in-out; border-color: @theme_color; } .author-box-wrap .td-author-name a, blockquote p, .page-nav a:hover, .widget_pages .current_page_item a, .widget_calendar td a, .widget_categories .current-cat > a, .widget_pages .current_page_parent > a, .td_pull_quote p { -webkit-transition: color 1s ease-in-out; -moz-transition: color 1s ease-in-out; -o-transition: color 1s ease-in-out; transition: color 1s ease-in-out; color: @theme_color; } .page-nav .current { -webkit-transition: background-color 1s ease-in-out; -moz-transition: background-color 1s ease-in-out; -o-transition: background-color 1s ease-in-out; transition: background-color 1s ease-in-out; background-color: @theme_color; border-color: @theme_color; } .wpb_btn-inverse, .ui-tabs-nav .ui-tabs-active a, .post .wpb_btn-danger, .form-submit input, .wpcf7-submit { -webkit-transition: background-color 1s ease-in-out; -moz-transition: background-color 1s ease-in-out; -o-transition: background-color 1s ease-in-out; transition: background-color 1s ease-in-out; background-color: @theme_color !important; } .header-search-wrap .dropdown-menu:before { -webkit-transition: border-color 1s ease-in-out; -moz-transition: border-color 1s ease-in-out; -o-transition: border-color 1s ease-in-out; transition: border-color 1s ease-in-out; border-color: transparent transparent @theme_color; } .td-mobile-content .current-menu-item > a, .td-mobile-content a:hover { -webkit-transition: color 1s ease-in-out; -moz-transition: color 1s ease-in-out; -o-transition: color 1s ease-in-out; transition: color 1s ease-in-out; color: @theme_color !important; } /* @slider_text */ .td-sbig-title-wrap .td-sbig-title, .td-slide-item-sec .td-sbig-title-wrap { -webkit-transition: background-color 1s ease-in-out; -moz-transition: background-color 1s ease-in-out; -o-transition: background-color 1s ease-in-out; transition: background-color 1s ease-in-out; background-color: @slider_text; } /* @select_color */ ::-moz-selection { background: @select_color; color: #fff; } ::selection { background: @select_color; color: #fff; } /* @link_color */ a, .widget_recent_comments .recentcomments .url { -webkit-transition: color 1s ease-in-out; -moz-transition: color 1s ease-in-out; -o-transition: color 1s ease-in-out; transition: color 1s ease-in-out; color: @link_color; } .cur-sub-cat { -webkit-transition: color 1s ease-in-out; -moz-transition: color 1s ease-in-out; -o-transition: color 1s ease-in-out; transition: color 1s ease-in-out; color:@link_color !important; } /* @link_hover_color */ a:hover, .widget_recent_comments .recentcomments .url:hover { -webkit-transition: color 1s ease-in-out; -moz-transition: color 1s ease-in-out; -o-transition: color 1s ease-in-out; transition: color 1s ease-in-out; color: @link_hover_color; } </style>';


var td_bg_boxed_imgs = [
'http://demo.tagdiv.com/newspaper/bg_images/1.jpg',
'http://demo.tagdiv.com/newspaper/bg_images/2.jpg',
'http://demo.tagdiv.com/newspaper/bg_images/3.jpg'
];


var td_current_panel_stat = td_read_cookie('td_show_panel');
if (td_current_panel_stat == 'show' || td_current_panel_stat == null) {
    jQuery('.td-theme-settings-small').addClass('td-theme-settings-no-transition');
    jQuery('.td-theme-settings-small').removeClass('td-theme-settings-small');
}




/*  ----------------------------------------------------------------------------
 On load
 */
jQuery().ready(function() {

    //hide panel
    jQuery("#td-theme-set-hide").click(function(event) {
        event.preventDefault();
        event.stopPropagation();
        //hide
        td_create_cookie('td_show_panel', 'hide', 1);
        jQuery('#td-theme-settings').removeClass('td-theme-settings-no-transition');
        jQuery('#td-theme-settings').addClass('td-theme-settings-small');


        jQuery('.td-set-theme-style-link').removeClass('fadeInLeft');


    });


    //show panel
    jQuery("#td-theme-settings").click(function() {
        if (jQuery(this).hasClass('td-theme-settings-small')) {

            jQuery('.td-set-theme-style-link').addClass('animated_xlong fadeInLeft');

            //show full
            td_create_cookie('td_show_panel', 'show', 1);
            jQuery('.td-theme-settings-small').removeClass('td-theme-settings-small');
        }
    });





}); //end on load






/*  ----------------------------------------------------------------------------
    Support functions
 */
//add trim for ie8
if (!String.prototype.trim) {
    String.prototype.trim = function() {
        return this.replace(/^\s+|\s+$/g, '');
    };
}


function td_create_cookie(name, value, days) {
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        var expires = "; expires=" + date.toGMTString();
    } else 
        var expires = "";
    document.cookie = escape(name) + "=" + escape(value) + expires + "; path=/";
}

function td_read_cookie(name) {
    var nameEQ = escape(name) + "=";
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') 
            c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) == 0) 
            return unescape(c.substring(nameEQ.length, c.length));
    }
    return null;
}

function td_erase_cookie(name) {
    td_create_cookie(name, "", - 1);
}





/*  ----------------------------------------------------------------------------
    live css compiler @tagDiv 2013
 */

//the settings object
function td_customizer_setting () {
    this.name = '';
    this.value = '';
}

//the sections object
function td_customizer_css_section() {
    this.name = '';
    this.raw_css = '';
    this.compiled_css = '';
}

//css parser class
function td_custom_css_parser(raw_css) {
    this.raw_css = raw_css;
    this.settings = [];
    this.css_sections = [];
    this.style_element_id = '';
    this.compiled_css = ''; //compiled css
}

//load the settings
td_custom_css_parser.prototype.load_setting_raw = function(name, value) {
    if (this.get_css_section(name) === false) {
        var new_setting = new td_customizer_setting();
        new_setting.name = name;
        new_setting.value = value;
        this.settings.push(new_setting);
    } else {
        this.update_setting_value(name, value);
    }
};

//split the css in sections
td_custom_css_parser.prototype.split_into_sections = function() {
    //remove style wrapping
    this.raw_css = this.raw_css.replace(/<style>/gi, '');
    this.raw_css = this.raw_css.replace(/<\/style>/gi, '');

    this.raw_css = this.raw_css.trim();

    //explode the sections
    var css_splits = this.raw_css.split('/*');

    var containing_class = this;
    jQuery.each(css_splits, function(index, css_split) {
        var css_split_parts = css_split.split('*/');
        if (typeof css_split_parts[0] !== "undefined" && typeof css_split_parts[1] !== "undefined") {
            var new_css_section = new td_customizer_css_section();
            new_css_section.name = css_split_parts[0].trim();
            new_css_section.raw_css = css_split_parts[1].trim();
            containing_class.css_sections.push(new_css_section);
        }
    });
};


//get setting value
td_custom_css_parser.prototype.get_setting_value = function(name) {
    var tmpReturn = false;
    jQuery.each(this.settings, function(index, setting) {
        if (setting.name === name) {
            tmpReturn = setting.value;
            return false; //brake jquery each
        }
    });
    return tmpReturn;
};

//get setting value
td_custom_css_parser.prototype.update_setting_value = function(name, value) {
    jQuery.each(this.settings, function(index, setting) {
        if (setting.name === name) {
            setting.value = value;
            return false; //brake jquery each
        }
    });
};


//get css section
td_custom_css_parser.prototype.get_css_section = function(name) {
    var tmpReturn = false;
    jQuery.each(this.settings, function(index, setting) {
        if (setting.name === name) {
            tmpReturn = setting.value;
            return false; //brake jquery each
        }
    });
    return tmpReturn;
};

//compile each section
td_custom_css_parser.prototype.compile_sections = function() {
    if (typeof this.css_sections !== "undefined" && typeof this.settings !== "undefined") {
        var containing_class = this;
        //console.log('start');

        jQuery.each(this.css_sections, function(index, section) {
            jQuery.each(containing_class.settings, function(index, setting) {
                section.raw_css = str_replace("@" + setting.name, setting.value, section.raw_css);
            });
        });
    }
};

//compile the css
td_custom_css_parser.prototype.compile_css = function() {

    this.split_into_sections();
    this.compile_sections();

    var buffy = '';
    var containing_class = this;

    jQuery.each(this.css_sections, function(index, section) {
        if (section.raw_css !== '' && containing_class.get_setting_value(str_replace("@", '', section.name)) !== false) {
            buffy = buffy + section.raw_css;
        }
    });

    this.compiled_css = buffy;
    //alert(buffy);
};


//inject css
td_custom_css_parser.prototype.inject_css = function() {


    var td_style = document.createElement('style');
    td_style.type = 'text/css';
    td_style.innerHTML = this.compiled_css;
    td_style.setAttribute("id", "td_style_inject");
    if (this.style_element_id === '') {
        //new element


        jQuery('body').append(td_style);
        this.style_element_id = 'td_style_inject';
    } else {
        //update old
        jQuery('#td_style_inject').replaceWith(td_style);
    }
};







/*
var td_custom_css_parser = new td_custom_css_parser(td_style_buffer);
td_custom_css_parser.load_setting_raw('header_color', 'red');
td_custom_css_parser.load_setting_raw('header_line_color', 'blue');


td_custom_css_parser.load_setting_raw('link_color', 'pink');

//td_custom_css_parser.compile_css();
//td_custom_css_parser.inject_css();


td_custom_css_parser.load_setting_raw('header_color', 'white');
td_custom_css_parser.load_setting_raw('header_line_color', 'gray');
*/
//td_custom_css_parser.compile_css();
//td_custom_css_parser.inject_css();



//alert(td_custom_css_parser.get_setting_value('link_color'));


//console.log(td_custom_css_parser.settings);
//console.log(td_custom_css_parser.css_sections);


//td_custom_css_parser.load_setting_raw('test');



var pad = function(num, totalChars) {
    var pad = '0';
    num = num + '';
    while (num.length < totalChars) {
        num = pad + num;
    }
    return num;
};

// Ratio is between 0 and 1
var changeColor = function(color, ratio, darker) {
    // Trim trailing/leading whitespace
    color = color.replace(/^\s*|\s*$/, '');

    // Expand three-digit hex
    color = color.replace(
    /^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i,
    '#$1$1$2$2$3$3'
    );

    // Calculate ratio
    var difference = Math.round(ratio * 256) * (darker ? - 1 : 1),
    // Determine if input is RGB(A)
    rgb = color.match(new RegExp('^rgba?\\(\\s*' +
    '(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])' +
    '\\s*,\\s*' +
    '(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])' +
    '\\s*,\\s*' +
    '(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])' +
    '(?:\\s*,\\s*' +
    '(0|1|0?\\.\\d+))?' +
    '\\s*\\)$'
    , 'i')),
    alpha = !!rgb && rgb[4] != null ? rgb[4] : null,
    // Convert hex to decimal
    decimal = !!rgb ? [rgb[1], rgb[2], rgb[3]] : color.replace(
    /^#?([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])/i,
    function() {
        return parseInt(arguments[1], 16) + ',' +
        parseInt(arguments[2], 16) + ',' +
        parseInt(arguments[3], 16);
    }).split(/,/),
    returnValue;

    // Return RGB(A)
    return !!rgb ?
    'rgb' + (alpha !== null ? 'a' : '') + '(' +
    Math[darker ? 'max' : 'min'](
    parseInt(decimal[0], 10) + difference, darker ? 0 : 255
    ) + ', ' +
    Math[darker ? 'max' : 'min'](
    parseInt(decimal[1], 10) + difference, darker ? 0 : 255
    ) + ', ' +
    Math[darker ? 'max' : 'min'](
    parseInt(decimal[2], 10) + difference, darker ? 0 : 255
    ) +
    (alpha !== null ? ', ' + alpha : '') +
    ')' :
    // Return hex
    [
    '#',
    pad(Math[darker ? 'max' : 'min'](
    parseInt(decimal[0], 10) + difference, darker ? 0 : 255
    ).toString(16), 2),
    pad(Math[darker ? 'max' : 'min'](
    parseInt(decimal[1], 10) + difference, darker ? 0 : 255
    ).toString(16), 2),
    pad(Math[darker ? 'max' : 'min'](
    parseInt(decimal[2], 10) + difference, darker ? 0 : 255
    ).toString(16), 2)
    ].join('');
};
var lighterColor = function(color, ratio) {
    return changeColor(color, ratio, false);
};
var darkerColor = function(color, ratio) {
    return changeColor(color, ratio, true);
};









function td_convert_hex(hex, opacity) {
    var hex = hex.replace('#', '');
    var r = parseInt(hex.substring(0, 2), 16);
    var g = parseInt(hex.substring(2, 4), 16);
    var b = parseInt(hex.substring(4, 6), 16);

    var result = 'rgba(' + r + ',' + g + ',' + b + ',' + opacity / 100 + ')';
    return result;
}

function str_replace (search, replace, subject, count) {
    // http://kevin.vanzonneveld.net
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: Gabriel Paderni
    // +   improved by: Philip Peterson
    // +   improved by: Simon Willison (http://simonwillison.net)
    // +    revised by: Jonas Raoni Soares Silva (http://www.jsfromhell.com)
    // +   bugfixed by: Anton Ongson
    // +      input by: Onno Marsman
    // +   improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +    tweaked by: Onno Marsman
    // +      input by: Brett Zamir (http://brett-zamir.me)
    // +   bugfixed by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   input by: Oleg Eremeev
    // +   improved by: Brett Zamir (http://brett-zamir.me)
    // +   bugfixed by: Oleg Eremeev
    // %          note 1: The count parameter must be passed as a string in order
    // %          note 1:  to find a global variable in which the result will be given
    // *     example 1: str_replace(' ', '.', 'Kevin van Zonneveld');
    // *     returns 1: 'Kevin.van.Zonneveld'
    // *     example 2: str_replace(['{name}', 'l'], ['hello', 'm'], '{name}, lars');
    // *     returns 2: 'hemmo, mars'
    var i = 0,
    j = 0,
    temp = '',
    repl = '',
    sl = 0,
    fl = 0,
    f = [].concat(search),
    r = [].concat(replace),
    s = subject,
    ra = Object.prototype.toString.call(r) === '[object Array]',
    sa = Object.prototype.toString.call(s) === '[object Array]';
    s = [].concat(s);
    if (count) {
        this.window[count] = 0;
    }

    for (i = 0, sl = s.length; i < sl; i++) {
        if (s[i] === '') {
            continue;
        }
        for (j = 0, fl = f.length; j < fl; j++) {
            temp = s[i] + '';
            repl = ra ? (r[j] !== undefined ? r[j] : '') : r[0];
            s[i] = (temp).split(f[j]).join(repl);
            if (count && s[i] !== temp) {
                this.window[count] += (temp.length - s[i].length) / f[j].length;
            }
        }
    }
    return sa ? s : s[0];
}

'use strict';
/**
 * updates the view counter thru ajax
 */
var td_ajax_count = {

    td_get_views_counts_ajax : function td_get_views_counts_ajax (page_type, array_ids) {

        //what function to call based on page type
        var page_type_action = 'td_ajax_get_views'; //page_type = page
        if (page_type == "post") {
            page_type_action = 'td_ajax_update_views';
        }

        jQuery.ajax({
            type: 'POST',
            url: td_ajax_url,
            cache: true,

            data: {
                action: page_type_action,
                td_post_ids: array_ids
            },
            success: function(data, textStatus, XMLHttpRequest) {
                var td_ajax_post_counts = jQuery.parseJSON(data); //get the return dara

                //check the return var to be object
                if (td_ajax_post_counts instanceof Object) {
                    //alert('value is Object!');

                    //itinerate thru the object
                    jQuery.each(td_ajax_post_counts, function(id_post, value) {
                        //alert(id_post + ": " + value);

                        //this is the count placeholder in witch we write the post count
                        var current_post_count = ".td-nr-views-" + id_post;

                        jQuery(current_post_count).html(value);
                        //console.log(current_post_count + ': ' + value);
                    });
                }
            },
            error: function(MLHttpRequest, textStatus, errorThrown) {
                //console.log(errorThrown);
            }
        });

    }
}

/*
    td_video_playlist.js
    v1.1
 */

"use strict";
// jQuery(window).load(function() {//
jQuery().ready(function() {

    //click on a youtube movie
    jQuery('.td_click_video_youtube').click(function() {

        //this flag is check to see if to start the movie
        td_youtube_player.td_playlist_video_autoplay_youtube = 1;

        //add pause to playlist control
        td_playlist_general_functions.td_playlist_add_play_control('.td_youtube_control');

        //create  and play the clicked video
        var td_youtube_video = jQuery(this).attr("id").substring(3);
        if (td_youtube_video != '') {
            td_youtube_player.playVideo(td_youtube_video);
        }
    });



    //click on youtube play control
    jQuery('.td_youtube_control').click(function() {

        //click to play
        if (jQuery(this).hasClass('td-sp-video-play')) {
            //this is to enable video playing
            td_youtube_player.td_playlist_video_autoplay_youtube = 1;

            //play the video
            td_youtube_player.td_playlist_youtube_play_video();

        } else {

            //put pause to the player
            td_youtube_player.td_playlist_youtube_pause_video();
        }
    });



    //check for youtube wrapper and add api code to create the player
    if (jQuery('.td_wrapper_playlist_player_youtube').length > 0) {

        if (jQuery('.td_wrapper_playlist_player_youtube').data("autoplay") == "1") {
            td_youtube_player.td_playlist_video_autoplay_youtube = 1;
        }

        var first_video = jQuery('.td_wrapper_playlist_player_youtube').data('first-video');

        if (first_video != '') {
            td_youtube_player.td_playlist_id_youtube_video_running = first_video;

            td_youtube_player.playVideo(first_video);
        }
    }



    //check autoplay vimeo
    if (jQuery('.td_wrapper_playlist_player_vimeo').data("autoplay") == "1") {
        td_vimeo_playlist_obj.td_playlist_video_autoplay_vimeo = 1;
    }

    //click on a vimeo
    jQuery('.td_click_video_vimeo').click(function() {

        //this flag is check to see if to start the movie
        td_vimeo_playlist_obj.td_playlist_video_autoplay_vimeo = 1;

        //add pause to playlist control
        td_playlist_general_functions.td_playlist_add_play_control('.td_vimeo_control');

        //create  and play the clicked video
        td_vimeo_playlist_obj.create_player(jQuery(this).attr("id").substring(3));
    });





    //check for vimeo wrapper and add api code to create the player
    if (jQuery('.td_wrapper_playlist_player_vimeo').length > 0) {

        //add play to playlist control
        td_playlist_general_functions.td_playlist_add_play_control('.td_vimeo_control');

        //create the iframe with the video
        td_vimeo_playlist_obj.create_player(jQuery('.td_wrapper_playlist_player_vimeo').data("first-video"));
    }




    //click on youtube play control
    jQuery('.td_vimeo_control').click(function() {

        //click to play
        if (jQuery(this).hasClass('td-sp-video-play')) {
            //this is to enable video playing
            td_vimeo_playlist_obj.td_playlist_video_autoplay_vimeo = 1;

            //play the video
            td_vimeo_playlist_obj.td_playlisty_player_vimeo.api("play");

        } else {

            //put pause to the player
            td_vimeo_playlist_obj.td_playlisty_player_vimeo.api("pause");
        }
    });

});


var td_youtube_player = {
    td_yt_player: '',

    td_player_container: 'player_youtube',

    td_playlist_video_autoplay_youtube: 0,

    td_playlist_id_youtube_video_running: '',


    playVideo: function(videoId) {
        if (typeof(YT) == 'undefined' || typeof(YT.Player) == 'undefined') {
            window.onYouTubePlayerAPIReady = function() {
                td_youtube_player.loadPlayer(td_youtube_player.td_player_container, videoId);
            };
            jQuery.getScript('//www.youtube.com/player_api');
        } else {
            td_youtube_player.loadPlayer(td_youtube_player.td_player_container, videoId);
        }
    },


    loadPlayer: function(container, videoId) {
        //container is here in case we need to add multiple players on page
        td_youtube_player.td_playlist_id_youtube_video_running = videoId;

        var current_video_name = td_youtube_list_ids['td_' + td_youtube_player.td_playlist_id_youtube_video_running]['title'];
        var current_video_time = td_youtube_list_ids['td_' + td_youtube_player.td_playlist_id_youtube_video_running]['time'];

        //remove focus from all videos from playlist
        td_playlist_general_functions.td_video_playlist_remove_focused('.td_click_video_youtube');

        //add focus class on current playing video
        jQuery('#td_' + videoId).addClass('td_video_currently_playing');

        //ading the current video playing title and time to the control area
        jQuery('#td_current_video_play_title_youtube').html(current_video_name);
        jQuery('#td_current_video_play_time_youtube').html(current_video_time);

        td_youtube_player.td_yt_player = '';
        jQuery(".td_wrapper_playlist_player_youtube").html("<div id=" + td_youtube_player.td_player_container + "></div>");

        td_youtube_player.td_yt_player = new YT.Player(container, {
            //window.myPlayer = new YT.Player(container, {
            playerVars: {
                //modestbranding: 1,
                //rel: 0,
                //showinfo: 0,
                autoplay: td_youtube_player.td_playlist_video_autoplay_youtube
            },
            height: '100%',
            width: '100%',
            videoId: videoId,
            events: {
                'onReady': td_youtube_player.onPlayerReady,
                'onStateChange': td_youtube_player.onPlayerStateChange
            }
        });
    },


    onPlayerStateChange: function onPlayerStateChange(event) {
        if (event.data == YT.PlayerState.PLAYING) {

            //add pause to playlist control
            td_playlist_general_functions.td_playlist_add_pause_control('.td_youtube_control');

        } else if (event.data == YT.PlayerState.ENDED) {
            //video_events_js.on_stop('youtube');

            //add play to playlist control
            td_playlist_general_functions.td_playlist_add_play_control('.td_youtube_control');

            //if a video has ended then make auto play = 1; This is the case when the user set autoplay = 0 but start watching videos
            td_youtube_player.td_playlist_video_autoplay_youtube = 1;

            //get the next video
            var next_video_id = td_playlist_general_functions.td_playlist_choose_next_video([td_youtube_list_ids, td_youtube_player.td_playlist_id_youtube_video_running]);
            if (next_video_id != '') {
                td_youtube_player.playVideo(next_video_id);
            }

        } else if (YT.PlayerState.PAUSED) {
            //add play to playlist control
            td_playlist_general_functions.td_playlist_add_play_control('.td_youtube_control');
        }
    },

    td_playlist_youtube_stopVideo: function td_playlist_youtube_stopVideo() {
        td_youtube_player.td_yt_player.stopVideo();
    },

    td_playlist_youtube_play_video: function td_playlist_youtube_play_video() {
        if (td_detect.is_mobile_device) {
            //alert('mobile');
        } else {
            td_youtube_player.td_yt_player.playVideo();
        }
    },

    td_playlist_youtube_pause_video: function td_playlist_youtube_pause_video() {
        td_youtube_player.td_yt_player.pauseVideo();
    }
};





//VIMEO
var td_vimeo_playlist_obj = {

    current_video_playing : '',

    td_playlisty_player_vimeo: '',
    //a copy of the vimeo player : needed when playing or pausing the vimeo pleyer from the playlist control

    td_playlist_video_autoplay_vimeo: '',
    //autoplay

    create_player: function (video_id) {
        if (video_id != '') {

            var vimeo_iframe_autoplay = '';

            this.current_video_playing = video_id;

            //remove focus class
            td_playlist_general_functions.td_video_playlist_remove_focused('.td_click_video_vimeo');

            //add focus clas on play movie
            jQuery('#td_' + video_id).addClass('td_video_currently_playing');

            //put movie data to control box
            this.put_movie_data_to_control_box(video_id);

            //check autoplay
            if (this.td_playlist_video_autoplay_vimeo != 0) {
                vimeo_iframe_autoplay = '&autoplay=1';
            }


            jQuery('.td_wrapper_playlist_player_vimeo').html('');
            jQuery('.td_wrapper_playlist_player_vimeo').html('<iframe id="player_vimeo_1" src="//player.vimeo.com/video/' + video_id + '?api=1&player_id=player_vimeo_1' + vimeo_iframe_autoplay + '"  frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>'); //width="100%" height="100%"

            this.create_vimeo_object_player(jQuery);
        }

    },

    put_movie_data_to_control_box: function (video_id) {
        jQuery('#td_current_video_play_title_vimeo').html(td_vimeo_list_ids['td_' + video_id]['title']);
        jQuery('#td_current_video_play_time_vimeo').html(td_vimeo_list_ids['td_' + video_id]['time']);
    },

    create_vimeo_object_player : function ($) {
        var iframe = '';
        var player = '';

        iframe = $('#player_vimeo_1')[0];
        player = $f(iframe)
        //a copy of the vimeo player : needed when playing or pausing the vimeo pleyer from the playlist control
        this.td_playlisty_player_vimeo = player;

        // When the player is ready, add listeners for pause, finish, and playProgress
        player.addEvent('ready', function() {
            //status.text('ready');

            player.addEvent('play', td_vimeo_playlist_obj.onPlay);
            player.addEvent('pause', td_vimeo_playlist_obj.onPause);
            player.addEvent('finish', td_vimeo_playlist_obj.onFinish);
            player.addEvent('playProgress', td_vimeo_playlist_obj.onPlayProgress);
        });
    },

    onPlay : function onPlay(id) {
        td_playlist_general_functions.td_playlist_add_pause_control('.td_vimeo_control');

        td_vimeo_playlist_obj.td_playlist_video_autoplay_vimeo = 1;
    },

    onPause : function onPause(id) {
        td_playlist_general_functions.td_playlist_add_play_control('.td_vimeo_control');
    },

    onFinish : function onFinish(id) {
        //status.text('finished');

        //add play to playlist control
        td_playlist_general_functions.td_playlist_add_play_control('.td_vimeo_control');

        //if a video has ended then make auto play = 1; This is the case when the user set autoplay = 0 but start watching videos
        td_vimeo_playlist_obj.td_playlist_video_autoplay_vimeo = 1;

        if (td_detect.is_mobile_device && td_detect.is_android) {
            //alert('is android');
        } else {

            //get the next video
            var next_video_id = td_playlist_general_functions.td_playlist_choose_next_video([td_vimeo_list_ids, td_vimeo_playlist_obj.current_video_playing]);
            if (next_video_id != '') {
                td_vimeo_playlist_obj.create_player(next_video_id);
            }
        }
    },

    onPlayProgress : function onPlayProgress(data, id) {
        //status.text(data.seconds + 's played');
    }
};


//this object holds some functions used by both the youtube and vimeo
var td_playlist_general_functions = {
    td_video_playlist_remove_focused: function td_video_playlist_remove_focused(obj_class) {
        //remove focus class
        jQuery( obj_class).each(function() {
            jQuery(this).removeClass('td_video_currently_playing');
        });
    },


    /*
        parram_array = array [
        video_list,
        current_video_id_playing
        ]
        */
    td_playlist_choose_next_video: function td_playlist_choose_next_video(parram_array) {
        //alert('get next');

        var video_list = parram_array[0];
        var current_video_id_playing = 'td_' + parram_array[1];

        //get next video id
        var next_video_id = '';
        var found_current = '';
        for (var video in video_list) {
            if (found_current == 'found') {
                next_video_id = video;
                found_current = '';
                break; //found , now exit
            }
            if (video == current_video_id_playing) {
                found_current = 'found';
            }
        }

        //play the next video
        if (next_video_id != '') {

            //remove 'td_' from the beginning of the string if necessary
            if (next_video_id.substring(0, 3) == 'td_') {
                next_video_id = next_video_id.substring(3);
            }

            return next_video_id;
        }

        return '';
    },



    //add pause button playlist control
    td_playlist_add_pause_control: function td_playlist_add_pause_control(wrapper_class) {
        jQuery(wrapper_class).removeClass('td-sp-video-play').addClass('td-sp-video-pause');
    },

    //add play button playlist control
    td_playlist_add_play_control: function td_playlist_add_play_control(wrapper_class) {
        jQuery(wrapper_class).removeClass('td-sp-video-pause').addClass('td-sp-video-play');
    }
}


