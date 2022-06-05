
                jQuery(
                         function() {
        $(".newsdesk .newsTabs a").on("click", function(event) {
            event.preventDefault();
            var channelId = $(this).attr("id");
            // clear all other tabs with active status
            $(this).siblings().removeClass("active");
            $(this).addClass("active");
            // only show the clicked channel's content
            $(this).parent().siblings(".channel").each(function() {
                if ($(this).attr("id") === channelId) {
                    $(this).show();
                } else {
                    $(this).hide();
                }
            });
        });
    }

                );
                jQuery(
                           function() {
         "use strict";
         var MIN_ITEMS = 6,
             data = {
                caller_name : "persrecs-homepage",
                start : 0,
                count: 24,
                noIncludes: 1, // Dont need to include recs CSS + JS collection from Mayhem
                standards: "personal",
                ref_marker: "hm_rec_lm"};
          $('#lateload-recs-widget').load(
              "/widget/recommendations/recs",
              data,
              function( response, status, xhr ) {
                  if (status === "error") {
                      if ($('.rec_slide .rec_item').length < MIN_ITEMS) {
                          $('.rec_heading_wrapper').parents('.article').hide();
                      }
                  }
                  if ('csm' in window) {
                      csm.measure('csm_recs_delivered');
                  }
              }
          );
      }

                );
                jQuery(
                     function() { $('#sidebar').watchlistRibbon('.ribbonize'); }
                );
        