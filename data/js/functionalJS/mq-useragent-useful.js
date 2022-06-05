
(function() {
    var _count = 0,
      MOBILE_MAX_WIDTH = 678,
      SOCIAL_MEDIA_MAX_WIDTH = 700,
      TABLET_MAX_WIDTH = 768;
    var navbarInterval = setInterval(function() {
        if(window.jQuery) {
            (function() {
                clearInterval(navbarInterval);

                var ww = $(window).width(),
                previousSize,
                previousDisplayOptions,
                // Makes the newly open submenu be on focus
                jumpTo = function(elem){
                    setTimeout(function(){
                        var new_position = $(elem).offset();
                        $("html, body").animate({ scrollTop: new_position.top - 36 }, 500);
                        return false;
                    }, 300);
                },
                
                isMobile = {
                    Android: function() {return navigator.userAgent.match(/Android/i);},
                    BlackBerry: function() {return navigator.userAgent.match(/BlackBerry/i);},
                    iPhone: function() {return navigator.userAgent.match(/iPhone/i);},
                    iPad: function() {return navigator.userAgent.match(/iPad/i);},
                    iOS: function() {return navigator.userAgent.match(/iPhone|iPad|iPod/i);},
                    Opera: function() {return navigator.userAgent.match(/Opera Mini/i);},
                    Windows: function() {return navigator.userAgent.match(/IEMobile/i);},
                    any: function() {return (isMobile.Android() || isMobile.BlackBerry() || isMobile.iOS() || isMobile.Opera() || isMobile.Windows());}
                },
                openSubMenu = function(e) {
                    var href = $(this).attr('href'),
	                    ref = $(this).attr('ref'),
	                    isClosingMenu = $(this).parent("li").hasClass("hover"),
	                    isTouchScreen = $('#navigationBar').hasClass('touch'),
	                    elem;
                	
                    if (ww <= MOBILE_MAX_WIDTH) { /* Mobile size */                        
                        if(!href || href == '' || href == '#') {
                            $(this).parent("li").toggleClass("hover");
                            if(isClosingMenu) {
                                elem = $($(this).parent().parent());
                            } else {
                                elem = $('#'+ref);
                            }
                            if(elem) {// Focus on the just open submenu
                                jumpTo(elem);
                            }                        
                        }
                        e.stopPropagation();
                    } else if (isTouchScreen) { /* Not mobile sized but touchscreen (iPad) */
                    	// On touchscreen devices we need to simulate hover state with css
                    	if(!href || href == '' || href == '#') {
                    		
                        	if($(this).parent('li').hasClass('hover')) {
                        		// Item already has hover, so just toggle
                        		$(this).parent("li").toggleClass("hover");
                        	} else {
                        		// Close all open dropdowns on the same level
                            	$(this).parent().parent('ul').children('.hover').removeClass('hover');
                                $(this).parent("li").toggleClass("hover");	
                        	}                        	
                        }
                        e.stopPropagation();                    	
                    }
                }, 
                isMobileSized = function(){
                  
                  if(ww <= MOBILE_MAX_WIDTH || isMobile.iPhone()){
                    return true;
                  }else{
                    return false;
                  }
                },
                
                isTabletSized = function(){
                  
                  if(ww <= TABLET_MAX_WIDTH || isMobile.iPad()){
                    return true;
                    
                  }else{
                    
                    return false;
                  }
                  
                },
                
                isSmallTabletSized = function(){
                  return (ww <= SOCIAL_MEDIA_MAX_WIDTH); 
                }, 
                
                adjustMenu = function() {
                  
                    if (isMobileSized()) { /* Mobile size */
                        
                        $(".toggleMenu").css("display", "inline-block");
                        if (!$(".toggleMenu").hasClass("active")) {
                            $(".nav").hide();
                        } else {
                            $(".nav").css('display', 'block');
                        }
                        
                        var logoPxsFromRight = ($(window).width() / 2) - ($('.navbarlogo').width() / 2)
                        $('.navbarlogo').css('right', logoPxsFromRight+'px');    
                        if(!jQuery.jGestures) {
                            head = document.getElementsByTagName('head')[0];                            
                            if(head){
                                script = document.createElement('script');
                                script.src =  "http://content.mqcdn.com/winston-624/cdn/jgestures/js/jgestures.min.js";
                                head.appendChild(script);
                            }
                        }
                        setSizeClassName('mobile-size');
                    
                    }else {
                    
                        $(".toggleMenu").css("display", "none");
                        $(".nav").css('display', 'inline-block');
                        $(".nav li").removeClass("hover");
                        
                        
                        if (isSmallTabletSized()) {
                          
                            setSizeClassName('tablet-size', { display: 'small'});
                        
                        }else if(isTabletSized()){
                          
                          setSizeClassName('tablet-size');
                          
                        }else {
                          
                            setSizeClassName('desktop-size');
                        }
                    }
                },
                
                fireEvent = function(value, options) {
                    var e = $.Event('navbar-boundary-changed', { size: value, options: options });
                    $('body').trigger(e);
                },
                
                setSizeClassName = function(className, options) {
                    if(className != previousSize || options != previousDisplayOptions) {
                        // Fire event just if boundary has been changed 
                        fireEvent(className, options);
                    }            
                    previousSize = className;
                    previousDisplayOptions = options;
                    $("#navigationBar").removeClass('desktop-size tablet-size mobile-size').addClass(className);
                    if(className == 'mobile-size') {
                        $('.navigationBarLinks').addClass('background_nav');
                        
                    } else { 
                        $('a.subMenuLink, a.sideMenuLink').unbind('click');
                        $('.navigationBarLinks').removeClass('background_nav');
                        $('.subMenu').attr('style', ''); // Remove styles set specific for mobile (if any)
                        $('.sideMenuMobile').css('display', 'none'); // Collapse all sideMenus opened (if any)
                        if($('.navbar-search-icon').hasClass('selected')) {
                            // Hide search bar if open 
                            $('.navbar-search-icon').toggleClass('selected');
                            $('#navigationBar .navbar-search').css('display', 'none');
                        }
                    }
                },
                
                toggleSearchBox = function() {
                    $('.navbar-search-icon').toggleClass('selected');
                    $('#navigationBar .navbar-search').slideToggle('fast');
                };                    
                
                
                $(".nav li a[id!='signin-link']").each(function() {
                    if ($(this).next().length > 0) {
                        $(this).addClass("parent");
                    };
                })
                
                $(".toggleMenu").click(function(e) {
                    e.preventDefault();
                    $(this).toggleClass("active");
                    $(".nav").slideToggle('fast');
                });

                $('.navbar-search-icon').click(function(event) {
                    event.stopPropagation();
                    toggleSearchBox();
                });
                
                $(window).bind('resize orientationchange', function() {
                    ww = document.body.clientWidth;
                    adjustMenu();
                });
                
                // Identify if the browser has touch-screen support
                if (!("ontouchstart" in document.documentElement)) {
                    $('#navigationBar').addClass("no-touch");
                } else {
                	$('#navigationBar').addClass("touch");
                }
                
                var eventType = isMobile.any() === true ? 'tapone' : 'click';
                $('#navigationBar').on(eventType, '.navigationBarLinks li', function(e) {
                    if($('#navigationBar').hasClass('mobile-size')) {
                        var linkTag = $(this).children().filter('a')[0];
                        e.stopPropagation();
                        // simulating click that works in iPhone as well
                        var click_ev = document.createEvent("MouseEvent");
                        click_ev.initEvent("click", true, true);
                        linkTag.dispatchEvent(click_ev);
                    }
                });
                
                $('#navigationBar')
                    .on('click', '.nav li a', openSubMenu)
                    .on('hover', '.nav .sideMenu', function (e) {
                        var parent = $(this).parent();
                        
                        parent.addClass('hover');
                    })
                    .on('mouseleave', '.nav .sideMenu', function (e) {
                        var parent = $(this).parent();
                        
                        parent.removeClass('hover');
                    });
                
                $('#navigationBar').on('click', 'a.subMenuLink, a.sideMenuLink', function() {
                    if (ww <= MOBILE_MAX_WIDTH) { /* Mobile size */
                        $($($(this).parent()).children().filter('ul')[0]).slideToggle('fast');
                    }
                });
                
                adjustMenu();
                // Hides address bar on iPhone
                window.scrollTo(0,1);
                // Avoid non-responsive navbar flashing before the mobile navbar kicks in
                var navbar = document.getElementById('navigationBar');
                navbar.style.display = 'block';
                
                // Script to lazy load Send2Cell widget
                var navigationBar = document.getElementById('navigationBar'),
                    links = navigationBar.getElementsByTagName('a'),
                    done = false,
                    i, isSend2Cell, send2CellTitle, send2CellMessage, script, head,
    
                loadS2CWidget = function(mobileLink, send2CellTitle, send2CellMessage){
                    window.mc.s2c.id(11).key("ao1kb7Gw0lAN_9_f")
                        .title(send2CellTitle)
                        .titleColor('#009ade')
                        .titleFontSize('16px')
                        .message(send2CellMessage)
                        .linkUrl(mobileLink.getAttribute('href')) /* link */
                        .background("url(http://content.mqcdn.com/winston-624/cdn/dotcom3/images/navbar_s2c_main_bg.jpg) no-repeat")
                        .backgroundSent("url(http://content.mqcdn.com/winston-624/cdn/dotcom3/images/navbar_s2c_sent_bg.jpg) no-repeat")
                        .roundIcon("false")
                        .linkColor("#009ADE")
                        .textColor("434343")
                        .sendColorLight("#8CC63F")
                        .sendColorDark("#638e2a")
                        .show();                        
                },
    
                onClickEventFactory = function(mobileLink, send2CellTitle, send2CellMessage){
                    return function(event){
                        if(!window.jQuery){
                            return true;
                        }
                        // Dont open widget when in mobile version
                        if($('#navigationBar').hasClass('mobile-size')) {
                            return true;
                        }
                        
                        if(window.mc && window.mc.s2c){
                            loadS2CWidget(mobileLink, send2CellTitle, send2CellMessage);
                        } else {
                            head = document.getElementsByTagName('head')[0];
    
                            if(head){
                                script = document.createElement('script');
                                script.src = "http://content.mqcdn.com/winston-624/cdn/loader.js.pre$locale=en_US&profile=send2cell";
                                script.onload = script.onreadystatechange = function(){
                                    if(!done && (!this.readyState || this.readyState === "loaded" || this.readyState === "complete")){
                                        done = true;
                                        window.setTimeout(function(){
                                            loadS2CWidget(mobileLink, send2CellTitle, send2CellMessage);
                                        }, 500);
    
                                        // Handle memory leak in IE
                                        script.onload = script.onreadystatechange = null;
                                        if (head && script.parentNode) {
                                            head.removeChild(script);
                                        }
                                    }
                                };
    
                                head.appendChild(script);
                            }
                        }
                        event = event || window.event;
                        if(event.preventDefault) {
                            event.preventDefault();
                        } else {
                            event.returnValue = false;
                        }
                        return false;
                    };
                };

                /* FIXME: There is  no better way of selecting mobile
                          linke until we add ids to navbar elements
                */
    
                for(i = 0; i < links.length; i++){
                    send2CellTitle = links[i].getAttribute('send2CellTitle');
                    send2CellMessage = links[i].getAttribute('send2CellMessage');
                    isSend2Cell = links[i].getAttribute('send2cell');
    
                    if(isSend2Cell && isSend2Cell === 'true' && send2CellTitle && send2CellTitle !== 'null' && send2CellMessage && send2CellMessage !== 'null'){
                        links[i].onclick = onClickEventFactory(links[i], send2CellTitle, send2CellMessage);
                    }
                }                    
                
            })();            
        }
        
        _count += 1;
        if(_count > 20) {
            var navbar = document.getElementById('navigationBar');
            navbar.className += ' desktop-size no-touch';
            navbar.style.display = 'block';
            clearInterval(navbarInterval);
        }
    }, 100);
})();
		  