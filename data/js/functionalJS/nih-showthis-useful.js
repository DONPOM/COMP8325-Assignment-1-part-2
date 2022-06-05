
	$(document).ready(function(){
		$("#globalnav ul li").hoverIntent({
			interval: 150, 
			over: drops_show, 
			timeout: 150, 
			out: drops_hide
		});
		$("#globalnav ul li").addClass('with-js');
	});
	function drops_show(){ $(this).addClass('show'); $(this).removeClass('with-js'); }
	function drops_hide(){ $(this).removeClass('show'); $(this).addClass('with-js'); }
