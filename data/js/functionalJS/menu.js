(function($){
    window.PBS = window.PBS || {};
    PBS.passportMenu = PBS.passportMenu || {};
    PBS.passportMenu.swipeIgnore = PBS.passportMenu.swipeIgnore || [];

    //get the schema set by Station Theme smart snippet on the body
    var BENTO_SCHEMA = $('body').attr('data-schema') ? '.'+$('body').attr('data-schema') : '';
    var nua = navigator.userAgent;
    var is_android = ((nua.indexOf('Mozilla/5.0') > -1 && nua.indexOf('Android ') > -1 && nua.indexOf('AppleWebKit') > -1) && !(nua.indexOf('Chrome') > -1));
    var is_safary  =(nua.indexOf('Safari') != -1 && nua.indexOf('Chrome') == -1);

    if(is_android){
        $('body').addClass('android');
    }

    if('ontouchstart' in document){
        $('body').addClass('no-hover');
    }else{
        $('body').removeClass('no-hover');
    }

    function isMobile(){
        return $('body').outerWidth() < 960;
    }

    function isPassportTheme(){
        return BENTO_SCHEMA.toLowerCase().indexOf('passport') != -1;
    }


    function isHorizSwipeIgnored(elem, ignoreArray){
        ignoreArray = ignoreArray || PBS.passportMenu.swipeIgnore;

        elem = $(elem);
        for(var i=0; i < ignoreArray.length; i++){
            if(elem.is(ignoreArray[i]) ||
                elem.closest(ignoreArray[i]).length){
                return true;
            }
        }

        return false;
    }

    function activateMenuButton(){
        if(!$.cookie("menu_active_item")){
            var l = window.location.href.toString().toLowerCase();
            l = l.substr(l.indexOf(window.location.host) + window.location.host.length, l.length);
            l = l.substr(0, l.indexOf("?") != -1 ? l.indexOf("?") : l.length);
            l = l.substr(0, l.indexOf("#") != -1 ? l.indexOf("?") : l.length);

            $('.nav.clearfix > li').removeClass('active');
            $('.nav').find('a[href="'+l+'"]').parents('.nav.clearfix > li:eq(0)').addClass('active');
        }else{
            $('.nav.clearfix > li').removeClass('active');
            $('.nav.clearfix > li:eq('+$.cookie("menu_active_item")+')').addClass('active');
            $.cookie("menu_active_item", '',{ expires: -1, path: '/' , domain: 'pbs.org'});
        }
    }

    function clearOpen(elems){
        elems.removeClass('open');
    }

    function fitNavigation(){
        if(!isPassportTheme()){
            return
        }

        var navContainer = $('.explorer'+BENTO_SCHEMA+' .nav-collapse ul.nav').closest('.container');
        var navBar = $('.explorer'+BENTO_SCHEMA+' .nav-collapse ul.nav');
        var menuItems = $('.explorer'+BENTO_SCHEMA+' .nav-collapse ul.nav > li').not(':empty');
        var menuSeparators = $('.explorer'+BENTO_SCHEMA+' .nav-collapse ul.nav > li:empty');

        if(isMobile()){
            if(isPassportTheme()){
                var nav = $('.nav-collapse.collapse');
                nav.appendTo('body');
                nav.find('.shadow').remove();
                nav.append('<div class="shadow"></div>');

                // var menu = $('.nav-collapse.collapse');
                // menu.find('li > .dropdownBg').unbind('click').bind('click', function(){
                //     // clearOpen(menu.find('li > .dropdownBg'));
                //     // $(this).closest('li').addClass('open');
                // })

                if(!$('.explorer'+BENTO_SCHEMA+' > .fake-bg').length){
                    $('.explorer'+BENTO_SCHEMA).append('<div class="fake-bg"></div>');
                    // $('.explorer'+BENTO_SCHEMA+' .fake-bg').css("height",$('body').height());
                }
            }
                
            menuItems.attr("style", "");
            navBar.attr("style", "");
        }else{
            if(!$('.navbar-inner > .nav-collapse.collapse').length){
                $('.nav-collapse.collapse').appendTo('.navbar-inner');
            }
            
            //negative right margin on LI makes UL wider than it's children
            var menuItemsWidth = 0;
            menuItems.each(function(){
                menuItemsWidth += $(this).outerWidth();
            });

            var menuSeparatorsWidth = 0;
            menuSeparators.each(function(){
                menuSeparatorsWidth += $(this).outerWidth();
            });
            if(!$.browser.msie){
                var diff = navContainer.width() - menuItemsWidth ;
            }else{
                var diff = navContainer.width() - menuItemsWidth;
            }

            var padding = parseInt(menuItems.css('padding-left'));
            var newPadding = (padding + diff / menuItems.length /2) > 0 ? (padding + diff / menuItems.length /2): 1;
            newPadding = Math.round(newPadding);
            menuItems.css("cssText", 'padding-left:'+newPadding +'px !important;'+'padding-right:'+newPadding +'px !important');
            // menuItems.closest('ul').width(menuItemsWidth+menuSeparatorsWidth)
             if(padding + diff / menuItems.length /2 < 0 || isMobile()){
                navBar.css("white-space", "normal");
                navBar.css("width", "auto");
                menuItems.css("cssText", "");


                $('.explorer'+BENTO_SCHEMA+' header div.main-menu .nav-collapse.collapse').show();

             }else{
                navBar.css("white-space", "nowrap");
             }
        }
    }

    function addTouchHandlers(){
        if(!isPassportTheme()){
            return;
        }
        if(!isMobile()){
            $(document).on("click", function(event){
                if(!$(event.target).parents('li.open .dropdown-menu').length){
                   $('.explorer'+BENTO_SCHEMA+' header div.main-menu .navbar li').removeClass('open')
                }
            })
            return;
        }
        var initialY = 0;
        var initialX = 0;
        var currentY = 0;
        var currentX = 0;
        var initialScroll = $(window).scrollTop();
        var windowH = $(window).height();
        var menuMaxTop = 0;
        var menuMinTop = 0;
        var insideMenu = false;


        $(document).on("touchstart", function(event){
            if(!isMobile()){
                return;
            }

            if(isHorizSwipeIgnored($(event.target))){
                return
            }

            insideMenu = false;
            if($(event.target).closest('.nav-collapse.collapse').length){
                insideMenu = true;
            }

            if(event.originalEvent.touches.length > 1){
                event.preventDefault();
            }

            menuMinTop = windowH - $('.nav-collapse.collapse ul').outerHeight() > 0 ?
                        0 :
                        windowH - $('.nav-collapse.collapse ul').outerHeight();

            initialY = event.originalEvent.touches[0].clientY;
            initialX = event.originalEvent.touches[0].clientX;
            initialScroll = $('.nav-collapse.collapse').offset().top;
            
        });

        $(document).on('touchmove', function(event) {
            if(!isMobile()){
                return;
            }
            if(isHorizSwipeIgnored($(event.target))){
                return
            }
            if(event.originalEvent.touches.length > 1){
                event.preventDefault();
            }

            currentY = event.originalEvent.touches[0].clientY;
            currentX = event.originalEvent.touches[0].clientX;

            var xDiff = Math.abs(currentX - initialX);
            var yDiff = Math.abs(currentY - initialY);

            if(xDiff >= yDiff){
                //unser wants to swipe horizontally
                event.preventDefault();
            }

            if($('body').hasClass('in')){
                if(isPassportTheme()){
                    event.preventDefault();       
                }
            }


        });

        $(document).on("touchend", function(event){
            

            if(isHorizSwipeIgnored($(event.target))){
                return
            }

            if(event.originalEvent.touches.length > 1){
                event.preventDefault();
            }
            var windowW = $(window).width();
            var diffX = currentX - initialX;
            var procentX = diffX * 100 / windowW;
            var diffY = currentY - initialY;
            var procentY = diffY * 100 / Math.abs($(window).outerHeight());
            var xTransition = $('.nav-collapse.collapse').css('-webkit-transform') || $('.nav-collapse.collapse').css('transform');
            xTransition = xTransition.split(",");
            xTransition = xTransition[xTransition.length-2];

            //Horizontal swipe
            if(Math.abs(procentX) > 30 && currentX){
                if(procentX > 0){
                    if(!$('body').hasClass('in')){
                        $('body').addClass('in');
                        // $('body').children().addClass('in');
                    }
                }else{
                    if($('body').hasClass('in')){
                        $('body').removeClass('in');
                        if(isPassportTheme()){
                            clearOpen($('.nav li'))
                        }
                    }
                }
            }

            if(Math.abs(procentY) > 20 && currentY){

                if(insideMenu){
                    if(isPassportTheme()){
                        var newTop = initialScroll + diffY;
                        var viewportHeight = $(window).height();
                        if(newTop > 0){
                            newTop = 0;
                        }
                        if(newTop < menuMinTop){
                            newTop = menuMinTop;
                        }
                        $('.nav-collapse.collapse').css('top', newTop);
                        event.preventDefault();
                    }
                }
            }


            initialY = 0;
            initialX = 0;
            currentY = 0;
            currentX = 0;
            initialScroll = $('.nav-collapse.collapse').offset().top;

        });
    }

    $(document).ready(function() {


        if($('.navbar').hasClass('split-button')){
            $('.split-button .nav-collapse .nav li a .caret').each(function(elem) {

               $(this).parent().after("<span class='dropdownBg' data-toggle='dropdown'><b class='caret'></b></span>");
               $(this).parent().removeAttr("data-toggle");
               $(this).remove();

            });
            $(".split-button .nav-collapse .nav li a").click(function(){
                var parentLi = $(this).parents('.nav.clearfix.btn-gpup > li');
                var parentIndex = $('.nav.clearfix.btn-gpup > li').index(parentLi);
                var url = $(this).attr("href");

                if (! /^https?:\/\/./.test(url) || new RegExp('https?://'+window.location.domain).test(url)){
                    //set cookie only if user stays on the domain
                    $.cookie("menu_active_item", '',{ expires: -1, path: '/' , domain: 'pbs.org'});
                    $.cookie("menu_active_item", parentIndex,{ expires: 300, path: '/' , domain: 'pbs.org'});
                }
                if($(this).attr("target") != "_blank"){
                    $('.explorer'+BENTO_SCHEMA).removeClass("in");
                    window.location = url;
                }
            });
        }else{
            $(".split-button .nav-collapse .nav li a").click(function(){
                if($(this).attr("target") != "_blank"){
                    $('.explorer'+BENTO_SCHEMA).removeClass("in");
                    // $('body').children().removeClass("in");
                }
            });
        }

        if(isPassportTheme()){
            $('.explorer'+BENTO_SCHEMA+' .navbar-inner .btn.btn-navbar').removeAttr('data-toggle');
            $('.explorer'+BENTO_SCHEMA+' .navbar-inner .btn.btn-navbar').click(function(){
                $('body').toggleClass("in");
                if(!$('body').hasClass('in')){
                    if(isPassportTheme()){
                        clearOpen($('.nav li'));
                    }
                }
            });
        }

        addTouchHandlers();

        fitNavigation();

        activateMenuButton();

        $('body').resize(function(){
            $('.explorer'+BENTO_SCHEMA+' .fake-bg').css("height",$('body').height());
        });

        if(is_safary && isPassportTheme()){
            //safary doesn't support onbeforeunload so we get all links to 
            //check when the user leaves the page and close the menu
            $('a').bind('click', function(){
                if($(this).attr('href') && $(this).attr('href').indexOf("#") !== 0){
                    $('body').removeClass('in');
                        clearOpen($('.nav li'))
                }
            })
        }
    });

    $(window).resize(function(){
        addTouchHandlers();

        fitNavigation();
        if(!isMobile()){
            $('.explorer'+BENTO_SCHEMA+' .navbar-inner ul.nav').css('top',"");
            $('.explorer'+BENTO_SCHEMA).removeClass("in");
            $('.explorer'+BENTO_SCHEMA+' .navbar-inner ul.nav .dropdown-menu').removeAttr("style");
            $('.nav-collapse.collapse').css('top', "");
            $(document).off("touchstart touchmove touchend");
        }

    });



    // $(window).load(function(){
    //     $('.explorer'+BENTO_SCHEMA+' .navbar-inner .fake-bg').css("height",$('body').height());
    // });

    $(window).on('beforeunload', function(){
        $('body').removeClass("in");
    })

    


}(jQuery));

