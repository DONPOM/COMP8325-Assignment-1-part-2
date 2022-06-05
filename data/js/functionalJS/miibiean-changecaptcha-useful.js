
	function changeCaptcha(){
		document.getElementById("vcode").src = '/getNavigationCode?' + Math.floor(Math.random()*100);
    }
	
	$('#vcode').click(function () { 
	    $(this).hide()
	      .attr('src', '/getNavigationCode?' + Math.floor(Math.random()*100)).fadeIn(); 
	  })

