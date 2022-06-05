
			jQuery(document).ready(function() {
                                // expand-content-link renamed to expand-cnt-link for compatibility with twentyfourteen theme
				jQuery(".expand-content-link").removeClass("expand-content-link").addClass("expand-cnt-link");
				jQuery(".expand-cnt-link").click(function() {
					jQuery(this).toggleClass("ecf_closed").parent(".exp-col-content-holder").find(".hidden-content").first().stop().slideToggle("slow").css("display","block");
					return false;
				});	
				jQuery(".expand-cnt-link").toggleClass("ecf_closed").parent(".exp-col-content-holder").find(".hidden-content").css("display","none");
			
			//images with no float styles , get floated left
			if(typeof jQuery(".hidden-content > img").attr("float") === "undefined") {
				jQuery(".hidden-content > img:not([class])").addClass("alignleft");
			}
			
			/*
			jQuery(".hidden-content").each(function() {
				if (jQuery(this).find("img").length) {
					var hiddenContentpLength = jQuery(this).find(".hiddenContentp").text().length;
						if( hiddenContentpLength < 200 ) {
							jQuery(this).css("height","150px");
						}
				}
			});
			*/
			
			jQuery(".textwidget > .exp-col-content-holder > .hidden-content > img+p").attr("style","display:inherit !important;");
			
			});
				