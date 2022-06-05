	//	VARIABLES DECLARATION
	
		//	Tabs and Accordion variables
			var cookieTabRunning = cookiePrefix+'tabRunning';
			//checkBackBox();
window.addEvent('domready', function() {
	
	//var inputBackBox  = new Element('input', {id: 'backbox', type: 'hidden', value: '0'});
	//inputBackBox.inject($('tabContainer'), 'bottom');

	initTabs();
	
	$$('.mediastabs li a').each(function(el){
		el.addEvent('click', function(){
			Cookie.write(cookieTabRunning, el.id, {path:"/"});
		});										 
	});	
	
});