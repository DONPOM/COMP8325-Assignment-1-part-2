
$(function(){
	$(window).load(function(){
		QQ.LoadScript("http://mat1.gtimg.com/www/js/common1.4.js",function(){
			setTimeout(function(){checkNonTxDomain(0.1, 11)}, 1000);
			setTimeout(function(){aqrjsCheckAllHtml(0.001, 0,11)},1000);
		});
	});
});
