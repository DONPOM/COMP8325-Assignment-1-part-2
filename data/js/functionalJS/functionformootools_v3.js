//bookmark

function CreateBookmarkLink(title,url) {

	if (window.sidebar) { // Mozilla Firefox Bookmark
		window.sidebar.addPanel(title, url,"");
	} else if( window.external ) { // IE Favorite
		window.external.AddFavorite( url, title); }
	else if(window.opera && window.print) { // Opera Hotlist
		return true; 
	}
}
//	SELECT REWRITING
function twAller(_url,_cadre,_target) {
  if (_cadre) {
    parent._cadre.location.href = _url;
  } else if (_target) {
    //_target = window.open(_url,_target);
    window.open(_url,_target);
  } else {
    window.location.href = _url;
  }
}
//	Write a select tag for this ul element tag
function buildTagSelect(id_ul,id_button,id_container) {

	var innerContent	= new Array;
	var innerValue		= new Array;
	var innerHref		= new Array;
	var selectTagHtml 	= '';
	
	var itemClassName	= $(id_ul).className;

	$$('#'+id_ul+' li').each(function(el){
						  
		var optionText	= '';
		var hrefExpr	= '';
		
		if (id_container == 'policies') {
			$(id_container).setStyle('height','46px');
		}
		
		if (el.className != 'displaynone') {
			
			if (Browser.Engine.trident){
				optionText 	= el.firstChild.innerHTML;
				hrefExpr	= el.firstChild.href;
			} else {
				optionText 	= el.childNodes[1].innerHTML;
				hrefExpr	= el.childNodes[1].href;
			}
		} else {
			hrefExpr	= '#';
			optionText	= el.innerHTML;
		}
		
		innerValue.include(optionText);
		innerHref.include(hrefExpr);
	});			
	
	innerContent 	= innerValue.associate(innerHref);
	var selectTag	= new Element('select', {
					'name': id_ul, 
					'id': id_ul
	});
	
	$$('#'+id_container+' .displaynone').each(function(el){
		el.erase('class');
	});
	
	if($(id_button))
		{
		$(id_button).getParent().erase('class');
		}
	
	each_index = 1;
	$each(innerContent, function(value, key) {
			
		var optionLine = '';
			
		optionLine += '<option';
		if ($chk(key)) {
		optionLine += ' value="'+key+'"';
		}
		
		
		if(each_index == $$('#'+id_ul+' li').length && id_ul == 'dropdown_policies'){
		optionLine += ' class="more"';
		}
		
		optionLine += '>';
		optionLine += value;
		optionLine += '</option>';
		
		selectTagHtml += optionLine;
		
		each_index ++;
	});
	
	selectTag.set('html', selectTagHtml);
	
	//	Write new element
	selectTag.replaces($(id_ul));
	selectTag.setStyle('display','inline');
	
	if($(id_button))
		{
		$(id_button).getParent('a').addEvent('click', function(el){
			location.href = selectTag.getSelected().getProperty('value');
		});
		}
}
//id_ul => id of the ul 
//action js action for on change event
function buildTagSelectSpecial(id_ul,action) {

	var innerContent	= new Array;
	var innerValue		= new Array;
	var innerHref		= new Array;
	var selectTagHtml 	= '';
	
	
	$$('#'+id_ul+' li').each(function(el){
						  
		var optionText	= '';
		var hrefExpr	= '';
		if (el.className != 'optiontitle') {
			
			if (Browser.Engine.trident){
	
					optionText 	= el.firstChild.innerHTML;
					hrefExpr	= el.firstChild.href;
					
					if(navigator.appVersion.indexOf('MSIE 9') != -1)
						{
						optionText 	= el.childNodes[1].innerHTML;
						hrefExpr	= el.childNodes[1].href;	
						}
					
			} else {
					optionText 	= el.childNodes[1].innerHTML;
					hrefExpr	= el.childNodes[1].href;
			}
		} else {
			hrefExpr	= '-1';
			optionText	= el.innerHTML;
		}
		
		innerValue.include(optionText);
		innerHref.include(hrefExpr);
	});			
	
	
	innerContent 	= innerValue.associate(innerHref);
	
	var selectTag	= new Element('select', {
					'name': id_ul, 
					'id': id_ul
	});
	
	
	each_index = 1;
	$each(innerContent, function(value, key) {
			
		var optionLine = '';
			
		optionLine += '<option';
		if ($chk(key)) {
		optionLine += ' value="'+key+'"';
		}
		
		
		optionLine += '>';
		optionLine += value;
		optionLine += '</option>';
		
		selectTagHtml += optionLine;
		
		each_index ++;
	});
	
	selectTag.set('html', selectTagHtml);
	
	//	Write new element
	selectTag.replaces($(id_ul));
	selectTag.setStyle('display','inline');
	
	if (action !=''){
		selectTag.addEvent('change', function(el){
			if (this.value != -1)
				eval(action);
		});
	}
	
}

//	TABS AND ACCORDION
//	History for tabs 



function checkBackBoxMootools() {
	
	if ($('backbox')){

		if ($('backbox').get('value') == 0) {
			//alert('NOT from browser history')
			//alert("egal  0: "+$('backbox').get('value'));
		} else {
			//alert('From browser history')
			//alert("egal  1: "+$('backbox').get('value'));
		}
		$('backbox').set('value', 1);
	}
}
function checkBackBox() {
	
	//if ($('backbox')){
	if (document.getElementById("backbox")){

		//if ($('backbox').get('value') == 0) {
		if (document.getElementById("backbox").value == 0) {
			//alert('NOT from browser history')
		} else {
			//alert('From browser history')
		}
		//$('backbox').set('value', 1);
		document.getElementById("backbox").value=1;
	}
}

//	Function who looks for the first item of a tab list
function isTabInList(tabRunning) {
	var tabInList	= false;
	$$('ul#mediabox_tabs li a').each(function(el) {	
		if(tabRunning == el.id) { tabInList = true; }
	});
	return tabInList;
}
//	Function who initialize the state of the running tabs and give action to the tabs link
function initTabs() {
	$$('.tabContent .content').setStyle('display','none');
	var tabRunning;
	var tabInList;

	//alert(document.referrer);
	
	//if ((document.referrer)) {
		
		tabRunning 	= Cookie.read(cookieTabRunning);
		tabInList	= isTabInList(tabRunning);
	//} 
	
	if ( (!tabRunning) || (!tabInList) ) {
		$$('ul#mediabox_tabs li:first-child a').each(function(el) {			   
			tabRunning = el.id;
		});
	}

	
	$$('#'+tabRunning).each(function(el) {
		if ($('tabContainer').hasChild('tabMenu')){
			initAccordionHome(el);	
		} else if ($('whatwasnew')||$('sitemapNav')){
			$('tabContainer').addClass('MultiAccordion');
			initAccordionMultiOpen(el);
		} else {
			
			initAccordion(el);		
		}
	});

	
	
	$$('.tabMenu a').each(function(el) {
		el.addEvent('click',function(e) {

			if ($('tabContainer').hasChild('tabMenu')){
				var ev = new Event(e).stop();
				initAccordionHome(el);
			} else {
				var ev = new Event(e).stop();
				
				if($('tabContainer').hasClass('MultiAccordion')) {
				
					initAccordionMultiOpen(el);
				}else{
					//if((!$('tabContainer').hasClass('noAccordion'))&&(navigator.appName.indexOf('Microsoft Internet Explorer') == -1)) {
					
					if((!$('tabContainer').hasClass('noAccordion'))&&(navigator.userAgent.indexOf('MSIE 7') == -1)) {
						myAccordion.display(0);
					}
					//initAccordion(el);
					initAccordion(el);
					//if(navigator.appName.indexOf('Microsoft Internet Explorer') != -1) {
					if(navigator.userAgent.indexOf('MSIE 7') != -1) {
						if(!$('tabContainer').hasClass('noAccordion')) {
							myAccordion.display(-1);
							//myAccordion.display(0);
							setTimeout("myAccordion.display(0)",300);
						}
					}
					
				
				//if(!$('tabContainer').hasClass('noAccordion')) {
					//setTimeout("adjustHeight(openedAcc)",300);
				//}
				}
			}
		});
		
	});
	
	
	//alert(window.location.href);
	
	
	
	
	//if ($('tabContainer')) {
		
		var colorName = '';
		colorName = searchColorRunning();
		//alert(colorName);
	
		$$('h5.toggler img').each(function(el){
			
			//alert(el.src.indexOf('plus'));
			if (el.src.indexOf('operator') != -1) {
				var suffix='operator-';
			}else{
				var suffix='';
			}
			
				if (el.src.indexOf('plus') != -1) {
					//
					el.src = '/wel/images/eu_portal/plus-'+suffix+colorName+'.gif';
				} else {
					el.src = '/wel/images/eu_portal/minus-'+suffix+colorName+'.gif';
				}
			
			
		});
	//}

}
function searchColorRunning() {
	
	var cookieStyleNameRunning 	= '';
	cookieStyleNameRunning 	= Cookie.read(cookieStyleName);
	var colorRunning	= 'blue';
	if (cookieStyleNameRunning) {
		colorRunning	= cookieStyleNameRunning.substring(stylesheetPrefix.length,cookieStyleNameRunning.length);
	}

	return colorRunning;
}
//	Function who initialize the accordion and choose randomly a tab to open
function initAccordion(thisEl) {

	tabState(thisEl);
	var maxItem = 0;
	$$('#mediabox_'+thisEl.id+'_content h5.toggler').each(function(el) {
		maxItem = maxItem + 1;														   
	});
	if (maxItem != 0) {
		// var randItem = 0;
		// if (thisEl.parentNode.parentNode.parentNode.id == 'tabMenu') {
			// randItem = $random(0, maxItem-1);
		// }
		var toggles = $$('#mediabox_'+thisEl.id+'_content h5.toggler');
		var content = $$('#mediabox_'+thisEl.id+'_content div.element');
		
		var colorName = '';
		
		myAccordion = new Accordion(toggles, content, {
			show:0,
			duration:200,
			display:0,	
			initialDisplayFx:false,
			onActive: function(toggler, element) {
				//openedAcc=element;
				colorName = searchColorRunning();
				
				if(toggler.getElement('img')) { toggler.getElement('img').src = '/wel/images/eu_portal/minus-'+colorName+'.gif'; }
			},
			onBackground: function(toggler, element){
				colorName = searchColorRunning();
				if(toggler.getElement('img')) { toggler.getElement('img').src = '/wel/images/eu_portal/plus-'+colorName+'.gif'; }	
			}
		});
	}
}
//	Function who initialize the accordion and choose randomly a tab to open
function initAccordionMultiOpen(thisEl) {
	
	tabState(thisEl);
	
	var maxItem = 0;
	$$('#mediabox_'+thisEl.id+'_content h5.toggler').each(function(el) {
		maxItem = maxItem + 1;														   
	});
	if (maxItem != 0) {
		
		
		
		
		var toggles = $$('#mediabox_'+thisEl.id+'_content h5.toggler');
		var content = $$('#mediabox_'+thisEl.id+'_content div.element');
		
		var collapsibles = new Array();
		
		var colorName = '';
		
		var openedToggle= Cookie.read('multiOpenAccordion');
		if (openedToggle==null){
			Cookie.write('multiOpenAccordion', ',,'+toggles[0].get('text').replace(/\s/g,''));
			var openedToggle=',,'+toggles[0].get('text').replace(/\s/g,'');
		}
		
	
		toggles.each(function(el,i) {
				
			var elmt=el.getNext('div.element');
			if (elmt!=null){
			
			var mySlide = new Fx.Slide(elmt);
			
			collapsibles[i] = mySlide;
			
			if (openedToggle.indexOf(','+el.get('text').replace(/\s/g,'')) == -1) {
			mySlide.hide();
			}else{
			el.getElement('img').src = '/wel/images/eu_portal/minus-operator-'+colorName+'.gif'; 
			}
			
			colorName = searchColorRunning();
			el.addEvent('click', function(e){
				var openedToggle= Cookie.read('multiOpenAccordion');
				if (mySlide.open){
					el.getElement('img').src = '/wel/images/eu_portal/plus-operator-'+colorName+'.gif';
					Cookie.write('multiOpenAccordion', openedToggle.replace(','+el.get('text').replace(/\s/g,''),''));
				}else{
					el.getElement('img').src = '/wel/images/eu_portal/minus-operator-'+colorName+'.gif'; 
					Cookie.write('multiOpenAccordion', openedToggle+','+el.get('text').replace(/\s/g,''));
				}
					mySlide.toggle();		
//alert(Cookie.read('multiOpenAccordion'))					
			});
			}
		});
	}
	
	if ($$('div.expandCollapse')[0]){
		$$('div.expandCollapse').setStyle('display','block');
	//expand collapse
		$('collapse_all_accordeon').addEvent('click', function(e){
			toggles.each( function(el, i) {
					if (collapsibles[i].open){
						var openedToggle= Cookie.read('multiOpenAccordion');
						collapsibles[i].toggle();
						el.getElement('img').src = '/wel/images/eu_portal/plus-operator-'+colorName+'.gif';
						Cookie.write('multiOpenAccordion', openedToggle.replace(','+el.get('text').replace(/\s/g,''),''));
					}
			});

		});
		$('expand_all_accordeon').addEvent('click', function(e){
			toggles.each( function(el, i) {
					if (!collapsibles[i].open){
						var openedToggle= Cookie.read('multiOpenAccordion');
						collapsibles[i].toggle();
						el.getElement('img').src = '/wel/images/eu_portal/minus-operator-'+colorName+'.gif'; 
						Cookie.write('multiOpenAccordion', openedToggle+','+el.get('text').replace(/\s/g,''));
					}
			});

		});
	}
}



//	Function who initialize the accordion and choose randomly a tab to open in the homepage
function initAccordionHome(thisEl) {
	
	tabState(thisEl);
	var maxItem = 0;
	$$('#mediabox_'+thisEl.id+'_content h5.toggler').each(function(el) {
		maxItem = maxItem + 1;														   
	});
	if (maxItem != 0) {
		var randItem = 0;
		randItem = $random(0, maxItem-1);
		
		var toggles = $$('#mediabox_'+thisEl.id+'_content h5.toggler');
		var content = $$('#mediabox_'+thisEl.id+'_content div.element');
		
		var colorName = '';
		//alert(colorName);
		myAccordion = new Accordion(toggles, content, {
			display:randItem,
			duration:200,
			onActive: function(toggler, element) {
				openedAcc=element;
				colorName = searchColorRunning();
				if(toggler.getElement('img')) { toggler.getElement('img').src = '/wel/images/eu_portal/minus-'+colorName+'.gif'; }
			},
			onBackground: function(toggler, element){
				colorName = searchColorRunning();
				if(toggler.getElement('img')) { toggler.getElement('img').src = '/wel/images/eu_portal/plus-'+colorName+'.gif'; }	
			}
		});
	}
}

//	Function who check which tab has to be open and close the others
function adjustHeight(obj){
	h=parseFloat(obj.getElement('div').scrollHeight)+7
	obj.setStyle('height',h+'px')
}

function tabState(ael) {

	$$('.tabMenu a').each(function(el) {
	
		if(el.hasClass('active')) {

			el.removeClass('active');
			
		}
	});
	ael.addClass('active');
	
	$$('.tabContent div.content').each(function(el) {
		
		if(el.hasClass('active')) {

			el.removeClass('active');
			el.setStyle('display','none');
		}
	});
	var pos_mediabox	= ael.getProperty('href').indexOf('mediabox',0);
	var href_property	= ael.getProperty('href').substring(pos_mediabox,ael.getProperty('href').length);
	//alert(ael.getProperty('href'));
	var ac = href_property;
	
	$(ac).addClass('active');
	$(ac).setStyle('display','block');

	
}
//	STYLESHEET SWITCHER
//	Function who initialize the stylesheet for the text and the color of the page
function initCss(cookieStyleName,cookieTextSize) {
	var cookieColor	= Cookie.read(cookieStyleName);
	var cookieText	= Cookie.read(cookieTextSize);
	
	if (cookieColor) {
		styleswitch('set',cookieColor,stylepath,cookieStyleName,"");
	}
	
	if (cookieText) {
		styleswitch('set',cookieText,stylepath,cookieTextSize,"");
	}
}
//	Function who switch a link to a css file by another
function styleswitch(mode,setstyle,stylepath,cookieName,onCompleteFct){

	/*alert(mode);*/
	//alert(setstyle);
	//alert(cookieName);

	
	$$('link').each(function(el){
		var itemplus	= el.href.split('.css');
		var posslash	= itemplus[0].lastIndexOf('/')+1;
		var namestyle	= itemplus[0].substring(posslash,itemplus[0].length);
		var position1 	= namestyle.indexOf('color', 0);
		var position2 	= namestyle.indexOf('text', 0);
		var position3 	= setstyle.indexOf('color', 0);
		var position4 	= setstyle.indexOf('text', 0);
		if ( ((position1 != -1) && (position3 != -1)) || ((position2 != -1) && (position4 != -1)) ) {
			el.href = stylepath+setstyle+'.css';
			Cookie.write(cookieName, setstyle, {duration:0, path:"/"});
		}	
	if (onCompleteFct){
		if(!$('tabContainer').hasClass('noAccordion')) {
			setTimeout(''+onCompleteFct+'',200);
		}
	}	
	});
	
	if ($('tabContainer')) {
	var colorName = searchColorRunning();
	//alert(colorName);
	
		$$('h5.toggler img').each(function(el){
			
			//alert(el.src.indexOf('plus'));
			if (el.src.indexOf('operator') != -1) {
				var suffix='operator-';
			}else{
				var suffix='';
			}
			if (el.src.indexOf('plus') != -1) {
				//
				el.src = '/wel/images/eu_portal/plus-'+suffix+colorName+'.gif';
			} else {
				el.src = '/wel/images/eu_portal/minus-'+suffix+colorName+'.gif';
			}
			
		});
	}
	
	
	if($('accessbar') && cookieName == 'europa_homepage_Stylesheet')
		{
		if(setstyle == 'stylesheetcolorblack')
			{
			//alert('hc');
			$('accessbar').getElements('img.sep').set('src', '/wel/images/accessbar_sep_high_contrast.gif');
			
			if($('dropdown_arrow'))
				{
				$('dropdown_arrow').set('src', '/wel/images/eu_portal/arrow-down3_high_contrast.gif');	
				}
			}
			else
			{
			//alert('normal');
			$('accessbar').getElements('img.sep').set('src', '/wel/images/accessbar_sep.gif');
			
			if($('dropdown_arrow'))
				{
				$('dropdown_arrow').set('src', '/wel/images/eu_portal/arrow-down3.gif');	
				}
			}
		}
		
		window.fireEvent('changeSize', 'ok');
}




//	Function who decompose an url to know where the current file is located in the folder tree
function lookForFolderIndentation(pageRunning,startToCount) {

	var infoPath;
	var arrayPageRunning	= pageRunning.split(".htm");
	var arrayUrl			= arrayPageRunning[0].split("http://");
	
	infoPath	= arrayUrl[1].split('/');
	var nbSubFolder	= infoPath.length-startToCount;
	var stringToMultiply	= '../';
	var stringPath = '';
	
	for(var i=0;i<nbSubFolder;i++) {
		stringPath += stringToMultiply;
	}
	
	return stringPath;
}
//	LANGUAGE SELECTOR
//	Function looked for the running language and the name of the running page
function lookForRunningInfo(pageRunning) {
	
	if (pageRunning.indexOf('.htm')<=0) pageRunning+='index_en.htm';	
	var lengthRunningUrl, nbCharCodeLang, nbCharPageExp, lookForLangStart, lookForPageStart, lookForPageEnd;
	var arrayPageRunning	= pageRunning.split(".htm");
	var runningInformation	= new Array;
	
	lengthRunningUrl		= arrayPageRunning[0].length;
	
	nbCharCodeLang			= 2;
	lookForLangStart		= lengthRunningUrl-nbCharCodeLang;
	
	nbCharPageExp			= 3;
	lookForPageStart		= arrayPageRunning[0].lastIndexOf('/',lengthRunningUrl)+1;
	lookForPageEnd			= lengthRunningUrl-nbCharPageExp;	
	
	runningInformation['lang'] = arrayPageRunning[0].substring(lookForLangStart,lengthRunningUrl);
	runningInformation['page'] = arrayPageRunning[0].substring(lookForPageStart,lookForPageEnd);
	
	return runningInformation;
}
//	Write a select tag for language selection
function makeSelectLang(runningInfo,ulIdLang,divIdLang,optionLang) {
	
	if(!$(ulIdLang)){return;}
	
	var innerLanguage		= new Array;
	var innerLangKey		= new Array;
	var innerLangValue	= new Array;
	var selectLang 			= '';	
	if ($(ulIdLang).title){labelLang=$(ulIdLang).title}else{labelLang=''};
	selectLang += '<label for="'+ulIdLang+'" class="displaynone">'+labelLang+'</label><select name="'+ulIdLang+'" id="'+ulIdLang+'" onChange="location=this.options[this.selectedIndex].value;">';

	$$('#'+ulIdLang+' li').each(function(el){
		
		var idLanguage	= el.id.replace(optionLang,'');
		var optionText	= el.childNodes[1].title +' ('+el.childNodes[1].innerHTML+') ';
		
		
		var attOptValue		= el.childNodes[1].href;
		if(navigator.appName.indexOf('Microsoft Internet Explorer') != -1) {
			var optionText	= el.firstChild.title +' ('+el.firstChild.innerHTML+') ';
			var attOptValue	= el.firstChild.href ;
		}
		
		/* IE9 */
		if(navigator.appName.indexOf('Microsoft Internet Explorer') != -1 && navigator.appVersion.indexOf('MSIE 9.0') != -1)
			{
			//alert(el.getElement('a').title);
			//return false;	
			if(el.getElement('a'))
				{
				var optionText	= el.getElement('a').title +' ('+el.getElement('a').innerHTML+') ';
				var attOptValue	= el.getElement('a').href 
				//alert(optionText)
				}
				else
				{
				var optionText	= el.getElement('span').title +' ('+el.getElement('span').innerHTML+') ';
				}
			}
		/* /IE9 */
			
		
		attOptSelected='';
		if (idLanguage == runningInfo['lang']) {

			attOptValue		= '#';
			attOptSelected	= ' selected="selected"';
		}
		
		selectLang += '<option id="'+idLanguage+'" value="'+attOptValue+'"'+attOptSelected+'>';
		selectLang += optionText;
		selectLang += '</option>';
	});	
	selectLang += '</select>';
	
	$(divIdLang).innerHTML = selectLang;
}


// validate FORM
function is_url(str)
	{
	var exp = new RegExp("^(http|https|ftp)\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(:[a-zA-Z0-9]*)?/?([a-zA-Z0-9\-\._\?\,\'/\\\+&amp;%\$#\=~])*$", "i");
	return exp.test(str);
	}
	

function testChamps(){
	$$('.errorField').each(function(el) {el.className=el.className.replace('errorField','')});
	var errorMsg='';
	var reg= /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]{2,}[.][a-zA-Z]{2,3}$/; 
	errorForm=false;	
	testEmptyField($('urls'));
	testEmptyField($('subjects'));
	testEmptyField($('comment'));
	testEmptyField($('e-mail'));
	
	var emptyTopic='';
	$$('input').each(function(el){
		if (el.getProperty('name')== 'Topic' && el.checked)
			{emptyTopic+=el.value;}						
	});
	
	if (emptyTopic.length==0){ 
		errorForm=true;
		$$('.subTdForm').each(function(el) {el.className='subTdForm errorField'});
	}
	if 	(errorForm){
		errorMsg+='<p>'+$('errorMsgEmpty').value+'</p>';		
	}
	
	if (checkemail($('e-mail').value)==null){
		errorMsg+='<p>'+$('errorMsgEmail').value+'</p>';		
	}
	
	if (is_url($('urls').value) == false){
		$('urls').className = 'buttons errorField';
		errorMsg+='<p>'+$('errorMsgURL').value+'</p>';		
	}
		
	if (errorMsg.length){
		$('error_messages').innerHTML=errorMsg;
		return false;
	}

}

function count_message(lekel, max_length)
	{
	current_ta_value = document.getElementById(lekel).value;
	
	if(current_ta_value.length > max_length)
		{
		document.getElementById(lekel).value = current_ta_value.substring(0, max_length);
		}
	}


function testEmptyField(obj){
	if (!obj.value.length){
		obj.className='buttons errorField';
		errorForm=true;
	}
}

function checkemail(email){
	var reg= /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]{2,}[.][a-zA-Z]{2,4}$/; 
	return (reg.exec(email))
	
	}
	
	
function isCssActivate(){
	
	var myContainer = new Element('div', {
		'id':'_iscss'
	});
	var myDiv_1 = new Element('div', {
		'id':'_iscss_1',
		'html':'div 1'
	});
	var myDiv_2 = new Element('div', {
		'id':'_iscss_2',
		'html':'div 2',
		'styles': {
			'width': '100px'
		}
	});
	
	myContainer.inject($('euLayout'), 'top');
	myDiv_1.inject(myContainer);
	myDiv_2.inject(myContainer, 'bottom');
	
	var boolOffSetWidth = myDiv_1.offsetWidth!=myDiv_2.offsetWidth;
	
	//alert(boolOffSetWidth);
	
	myContainer.dispose();
	//if (window.console) console.log('is css:'+boolOffSetWidth);
	
	
	
	return boolOffSetWidth;
}	


/*FVI For the survey*/
//include JS file	
function includeJS(jsPath,fileName){
	document.write('<script src="'+jsPath+fileName+'" type="text/javascript"></script>');
}

//include CSS file	
function includeCSS(cssPath,fileName){
	document.write('<link type="text/css" href="'+cssPath+fileName+'" rel="stylesheet"/>');
}
/*FVI For the survey*/