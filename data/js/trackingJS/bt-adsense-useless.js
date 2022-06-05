
          $(function() {
               $('#next-ads').css('display', '');
               $('#front-ads').after('<ul id="banner-nav">').cycle({
                    speed:  1000,           //speed of the transition (any valid jquery fx speed value) 
                    timeout: 5000,      //milliseconds between slide transitions
                    slideExpr: 'div.largefeature',
                    pager: '#banner-nav',
                    pauseOnPagerHover: true,
                    pause:  1, // true to enable "pause on hover" 
                    // callback fn that creates a link to use as the pager anchor 
                    pagerAnchorBuilder: function(idx, slide) { 
                         return '<li><a href="#">&bull;</a></li>'; 
                    }
               });
          });
          // redefine Cycle's updateActivePagerLink function 
          $.fn.cycle.updateActivePagerLink = function(pager, currSlideIndex) { 
               $(pager).find('li').removeClass('activeLI') 
               .filter('li:eq('+currSlideIndex+')').addClass('activeLI'); 
          };
          