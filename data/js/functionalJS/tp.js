$(document).ready(function() {	
    $('#loginLink').click(function() {
        // Allow everyone but IE6 to display the faded-in login box
        if ($.browser.version != '6.0') {
            $(".loginBOX").fadeIn(250);
            $("#username", ".loginBOX").focus();
            return false;
        }
    });
    
    // faq (article view) buttons
    $('.faq-btn').click(function() {
        var i = parseInt(this.id.replace("faqBtn", ""));
        $(".faq").stop().slideUp();
        $("#faq" + i).stop().slideDown();
        $('.faq-btn').removeClass('activeClass');
        $(this).addClass('activeClass');
        return false;
    });
    
    $('#viewAll').click(function(){
        $(".faq").stop().show();
    });
    

    // closes login box if user decides not to log in
    $('.logclose-btn').click(function() {
        $(".loginBOX").fadeOut(250);
        return false;
    });
    
    function swapImages() {
      var $active = $('#fadeShow .active');
      var $next = ($('#fadeShow .active').next().length > 0) ? $('#fadeShow .active').next() : $('#fadeShow li:first');
        $next.fadeIn().addClass('active');
        $active.fadeOut(function(){
        $active.removeClass('active');
      });
    }
    
    // Run our swapImages() function every 5secs
    setInterval(swapImages, 5000);
    
    // cpanel hide/show sidebar blocks  ----- check to see if used anywhere else on site. if not, remove - 3-15.2013
    $('.toggle-btn').click(function() {
        var i = parseInt(this.id.replace("toggleBtn", ""));
        var $me = $(this);
        var $desc = $("#toggleSide" + i);
        var isShow = !$me.hasClass("selected");

        // Stop Animations and roll up everyone.
        $(".toggle").stop(true, true).slideUp();
        $(".toggle-btn").removeClass("selected");

        if (isShow) {
            $me.addClass("selected");
            $desc.slideDown();
        }
        return false;
    }); // end cpanel hide/show
	
    
    // launch video
    $("#launchVideo").click(function(){
      $("#getToKnowIframe").attr("src", "http://player.vimeo.com/video/29624382?title=0&amp;byline=0&amp;portrait=0&amp;api=1");
          $("#getToKnow").fadeIn(400);
          return false;
    });
    
    $('.videoHolder').click(function(){
      $("#getToKnowIframe").attr("src", "");
        $('.videoHolder').fadeOut(300);
        return false;
    });
	
	// launch video 2
    $("#launchVideo").click(function(){
      $("#getToKnowIframe2").attr("src", "http://player.vimeo.com/video/32321710?title=0&amp;byline=0&amp;portrait=0&amp;api=1");
          $("#getToKnow").fadeIn(400);
          return false;
    });
    
    $('.videoHolder').click(function(){
      $("#getToKnowIframe2").attr("src", "");
        $('.videoHolder').fadeOut(300);
        return false;
    });
	
	// load modal for free signup
	$('.freeSignupBtn').click(function(){
		$('#freeBoxFrame').attr('src', function(){return $(this).data('src')});
		$('#freeBox').fadeIn();
		return false;
	});
	

	$('.free-close').click(function(){
		$('#freeBox').fadeOut();
		return false;
	});
	 
    // pricing page
    var urlTemplate = '/bin/membership/pref_link?to=buy&amp;m_ORIGIN=&amp;plan=%%PLAN%%&amp;view=iframe';
    var resetLabels = function(){
        $(".planGo").html("Sign Up Now!");
    }
    
    $('.planGo').click(function(){
        var i = $(this).attr("href").replace("#", "");
        var newURL = urlTemplate.replace('%%PLAN%%', i);
        $(this).html("Loading...");
        $('html, body').animate({ scrollTop: 0 }, 'slow');
        $("#windowbox-content").unbind()
            .load(function() {
                $('#step1').hide();
                $('#step2').show();
            })
            .attr("src", newURL);
    });
    
    $('#back').click(function(){
        resetLabels();
        $('#step2').hide();
        $('#step1').show();
        return false;
    }); 
    
	var slideAway = function(ev) {
		if (!$(ev.target).hasClass('priceTitle')) {
			$('.featureListings').slideUp();
			$('.exOut').hide();
		}
	}
	
    // hide / show pricing details
	$('.exOut').click(slideAway);
	    
	$(".priceTitle").click(function() {
		var $t = $(this).parents('tr').find('.featureListings');
		$('.exOut').hide();
		$(this).parents('tr').find('.exOut').show();
		var $not = $(this).parents('table').find('.featureListings').not($t);
		$t.slideToggle();
		$not.slideUp();
	});
	
	// END pricing js
    
});
