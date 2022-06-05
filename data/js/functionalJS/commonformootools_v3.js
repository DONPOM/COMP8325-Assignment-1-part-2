//document.getElementById('dropdown_contact_hq_v2').style = "display:none;";
//	VARIABLES DECLARATION

//	General Common variables
var cookiePrefix = 'europa_homepage_';
var dispClName = 'displayed';
var dispClIName = 'displayedIn';
var hiddClName = 'hidden';
var hiddClIName = 'hiddenIn';
var imgPath = '/wel/images/eu_portal/';

//	Language selection variables
var divIdLang = 'lgg';
var ulIdLang = 'language_selector';
var optionLang = 'lang_option_';
var pageRunning = document.location.href;

//	Print
var idPriText = 'printVersion';

// Text Resize
var idIncText = 'stylesheettexthigh';
var idMidText = 'stylesheettextmid';
var idDecText = 'stylesheettextlow';

if (navigator.userAgent.indexOf('MSIE 6') > 0) {
    var idIncTextCss = 'stylesheettexthigh_ie6';
    var idMidTextCss = 'stylesheettextmid_ie6';
    var idDecTextCss = 'stylesheettextlow_ie6';
} else {
    var idIncTextCss = 'stylesheettexthigh';
    var idMidTextCss = 'stylesheettextmid';
    var idDecTextCss = 'stylesheettextlow';
}

var cookieTextSize = cookiePrefix + 'sizeFont';

// Style switcher
var classStyle = 'styleSwitcher';
var cookieStyleName = cookiePrefix + 'Stylesheet';
var stylesheetPrefix = 'stylesheetcolor';

var openedAcc = '';

var stringIndent = lookForFolderIndentation(pageRunning, 5);
//var stylepath		= stringIndent+'/wel/stylesheets/eu_portal/';
var stylepath = '/wel/stylesheets/eu_portal/';
// Search field clear
var idSearchField = 'searchfield';


//document.write('<style>#dropdown_contact_hq_v2{display:none;}</style>');

document.write('<style>#dropdown_policies{display:none;}</style>');
document.write('<style>#dropdown_archives{display:none;}</style>');
document.write('<style>#dropdown_policy_areas{display:none;}</style>');
//document.write('<style>#dropdown_archives{display:none;}</style>');
document.write('<style>#lgg{display:none;}</style>');



document.write('<style>#accessbar_container.eu2col{width:100%; position:relative;}</style>');
document.write('<style>#accessbar.eu2col{margin-left:0px; right:10px; _right:8px;}</style>');
document.write('<style>#accessbar_container.eu3col{width:100%; position:relative;}</style>');
document.write('<style>#accessbar.eu3col{margin-left:0px; right:-215px; _right:-218px;}</style>');

document.write('<style>#stylesheettextlow img{margin-right:2px}</style>');
document.write('<style>#stylesheettextmid img{margin-right:2px}</style>');
document.write('<style>#stylesheettexthigh img{margin-right:2px}</style>');
document.write('<style>#stylesheetcolorblue img{margin-right:6px}</style>');
document.write('<style>#stylesheetcolorgreen img{margin-right:6px}</style>');
document.write('<style>#stylesheetcolorred img{margin-right:6px}</style>');
document.write('<style>#stylesheetcolorblack img{margin-right:6px}</style>');

/*FVI For the survey*/
includeCSS("/wel/stylesheets/eu_portal/", "smoothcss.css");
includeJS("/wel/scripts/eu_portal/", "cookies.js");
includeJS("/wel/scripts/eu_portal/", "smoothbox12.js");
includeJS("/europa_survey/scripts/", "europa_survey.js");

/*FVI For the survey*/

tab_index = 0;
function auto_rotate_tabs()
{
    //alert(tab_index);
    $$('ul#mediabox_tabs li')[0].getElement('a').removeClass('active');
    $$('ul#mediabox_tabs li')[1].getElement('a').removeClass('active');
    $$('ul#mediabox_tabs li')[2].getElement('a').removeClass('active');

    $$('div#tabContent div.content')[0].setStyle('display', 'none');
    $$('div#tabContent div.content')[0].removeClass('active');
    $$('div#tabContent div.content')[1].setStyle('display', 'none');
    $$('div#tabContent div.content')[1].removeClass('active');
    $$('div#tabContent div.content')[2].setStyle('display', 'none');
    $$('div#tabContent div.content')[2].removeClass('active');


    if (tab_index > 2)
    {
        tab_index = 0;
    }

    $$('ul#mediabox_tabs li')[tab_index].getElement('a').addClass('active');
    $$('div#tabContent div.content')[tab_index].setStyle('display', 'block');
    $$('div#tabContent div.content')[tab_index].addClass('active');
    tab_index ++;

    setTimeout("auto_rotate_tabs()", 5000);
}

window.addEvent('domready', function() {



    //Accessbar repositionning
    if ($('accessbar').get('class').indexOf('accessbar_hp') == - 1)
    {
        var mySecondElement = new Element('div', {
            id: 'accessbar_container'
        }).wraps('accessbar');
    }

    //alert($('accessbar').getParent().getParent().get('class'));

    current_context = $('accessbar').getParent().getParent().get('class');

    if ((current_context.indexOf('euContainer2col') != - 1 || current_context.indexOf('euContent2col') != - 1))
    {
        $('accessbar').addClass('eu2col');
        $('accessbar_container').addClass('eu2col');
    } else
    {
        if ($('accessbar').get('class').indexOf('accessbar_hp') == - 1)
        {
            $('accessbar').addClass('eu3col');
            $('accessbar_container').addClass('eu3col');
        }
    }

    if (isCssActivate() == true)
    {
        $('accessbar').setStyle('display', 'inline');
    }


    /* IE9 */
    if (navigator.appVersion.indexOf('Chrome') != - 1 || navigator.appVersion.indexOf('MSIE 9.0') != - 1 || navigator.appVersion.indexOf('Opera') != - 1)
    {
        if ($('accessbar').get('class').indexOf('accessbar_hp') != - 1)
        {
            var mySecondElement = new Element('div', {
                id: 'accessbar_container'
            }).wraps('accessbar');

            $('accessbar_container').setStyle('position', 'relative');

            $('accessbar').setStyle('right', '15px');
        }
    }
    /* /IE9 */


    if (navigator.userAgent.indexOf('Opera') != - 1)
    {
        if ($('accessbar').get('class').indexOf('accessbar_hp') != - 1)
        {
            $('accessbar').setStyle('margin-left', '654px');
        }
    }



    if (window.location.href.indexOf('copy2_index_en.htm') != - 1)
    {
        //alert('ici');
        auto_rotate_tabs();
    }

    // CCB
    var language_selector = $("lgg");
    if ( language_selector ) {
        $('lgg').setStyle('display', 'block');
    }

    /*alert(isCssActivate());
    		if ($('accessbar') && isCssActivate() == true) {
    			$('accessbar').setStyle('display','inline');
    		}*/

    //alert(cookieStyleName)
    //alert(cookieTextSize)
    initCss(cookieStyleName, '');
    //	SCRIPTING

    //	Display of Hidden Object
    //	All Block Elements
    $$('.' + hiddClName).each(function(el) {
        el.className = dispClName;
    });
    //	All Inline Elements
    $$('.' + hiddClIName).each(function(el) {
        el.className = dispClIName;
    });

    // CCB
    if ( ulIdLang )
    {
        //	Language selection
        //	Hide The Language Div Box
        $$('#' + divIdLang).each(function(el) {
            el.className = hiddClName;
        });
        //	Look for the current language in the URL
        var runningInformation = lookForRunningInfo(pageRunning);
        //	Writing of the select tag
        makeSelectLang(runningInformation, ulIdLang, divIdLang, optionLang);
        //	Display The Language Div Box
        $$('#' + divIdLang).each(function(el) {
            el.className = dispClName;
        });
    }

    //	Color switcher
    //
    if ($('accessbar')) {
        $$('.' + classStyle).each(function(el) {
            //el.set('href','javascript:void(0);');
            el.addEvent('click', function() {
                styleswitch('set', el.id, stylepath, cookieStyleName, "");
                return false;
            });
        });
    }

    //	Print
    // Print the page
    if ($('accessbar')) {
        //$(idPriText).set('href','javascript:void(0);');
        $$('#' + idPriText).addEvent('click', function(el) {
            self.print();
        });
    }

    //	Text Resize
    if ($('accessbar')) {
        //	Increase the text size
        //$(idIncText).set('href','javascript:void(0);');

        $(idIncText).addEvent('click', function(el) {
            argmt = "";
            if (openedAcc) {
                argmt = "adjustHeight(openedAcc)" 
            };
            styleswitch('set', idIncTextCss, stylepath, cookieTextSize, argmt);
            return false;
        });

        //	Decrease or increase the text size to medium size
        //$(idMidText).set('href','javascript:void(0);');
        $(idMidText).addEvent('click', function(el) {
            argmt = "";
            if (openedAcc) {
                argmt = "adjustHeight(openedAcc)" 
            };
            styleswitch('set', idMidTextCss, stylepath, cookieTextSize, argmt);
            return false;
        });

        //	Decrease the text size
        //$(idDecText).set('href','javascript:void(0);');
        $(idDecText).addEvent('click', function(el) {
            argmt = "";
            if (openedAcc) {
                argmt = "adjustHeight(openedAcc)" 
            };
            styleswitch('set', idDecTextCss, stylepath, cookieTextSize, argmt);
            return false;
        });
    }

    //	Drop Down Generation

    if ($('dropdown_contact_hq_v2')) {

        $('dropdown_contact_hq_v2').setStyle('position', 'absolute');
        $('dropdown_contact_hq_v2').setStyle('top', '-5000px');
        $('dropdown_contact_hq_v2').setStyle('display', 'block');
        $('dropdown_contact_hq_v2').setStyle('left', '629px');

        if ($('quicklinks_button_v2'))
        {
            $('quicklinks_button_v2').setStyle('display', 'inline');
        }

        // ------ DROP TITLE -------------------- --------------------------------------------------------------------


        $('euLayout').getElements('a[class!=ql_links]').addEvent('focus', function(el) {
            //alert('focus a');
            if ($('dropdown_contact_hq_v2'))
            {
                collapse('dropdown_contact_hq_v2', '');
            }
        });

        /*$('last_ql_links').addEvent('focus', function(el){
        					expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
        					})*/


        /*$('dropdown_contact_hq_v2').getElement('ul').getElement('.quick_links_more').getElement('a').setStyle('background-color', '#ff0000');
        				$('dropdown_contact_hq_v2').getElement('ul').getElement('.quick_links_more').getElement('a').addEvent('mouseover', function(el){
        					alert('focus');
        					$('dropdown_contact_hq_v2').setStyle('display','block');				
        					expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
        					});*/



        $('droptitle_contact_hq_v2').addEvent('mouseout', function(el) {
            //collapse('dropdown_contact_hq_v2','');	
            //$('dropdown_contact_hq_v2').setStyle('display','none');

        });


        if ($('selector_dropdown').getElement('label'))
        {
            $('selector_dropdown').getElement('label').addEvent('click', function(el) {
                if (navigator.appName == 'Microsoft Internet Explorer')
                {
                    $('dropdown_contact_hq_v2').setStyle('display', 'block');
                    expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
                }

            });
        }


        if ($('dropdown_arrow'))
        {
            $('dropdown_arrow').addEvent('click', function(el) {
                if (navigator.appName == 'Microsoft Internet Explorer')
                {
                    $('dropdown_contact_hq_v2').setStyle('display', 'block');
                    expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
                }

            });
        }


        $('droptitle_contact_hq_v2').addEvent('click', function(el) {
            $('dropdown_contact_hq_v2').setStyle('display', 'block');
            expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
            //$('quicklinks_button_v2').focus();
        });



        $('droptitle_contact_hq_v2').addEvent('keyup', function(el) {
            $('dropdown_contact_hq_v2').setStyle('display', 'block');
            expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
        });
        // ------- DROP DOWN ---------------------------------------- ------------------------------------------------
        $('dropdown_contact_hq_v2').addEvent('mouseout', function(el) {
            collapse('dropdown_contact_hq_v2', '');
            //$('dropdown_contact_hq_v2').setStyle('display','none');
        });



        $('dropdown_contact_hq_v2').addEvent('keyup', function(el) {
            if (el.key == "enter") {
                collapse('dropdown_contact_hq_v2', '');
                if ($('quicklinks_button_v2'))
                {
                    $('quicklinks_button_v2').focus();
                }

                $('dropdown_contact_hq_v2').setStyle('display', 'none');
            }
        });
        $('dropdown_contact_hq_v2').addEvent('mouseover', function(el) {
            $('dropdown_contact_hq_v2').setStyle('top', '-5000px');
            $('dropdown_contact_hq_v2').setStyle('display', 'block');
            expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
            if ($('quicklinks_button_v2'))
            {
                $('quicklinks_button_v2').focus();
            }
        });
        // -------------------------------------------------------------------- -----------------------------------------------
        $$('div#dropdown_contact_hq_v2 ul li').each(function(el) {
            el.title = el.getElement('a').href;
            //el.innerHTML = '<a href="javascript:void(0);">'+el.getElement('a').innerHTML;+'</a>';
            el.innerHTML = '<a class="ql_links" href="' + el.title + '" title="' + el.getElement('a').innerHTML + '">' + el.getElement('a').innerHTML;
            + '</a>';
            el.getElement('a').addEvent('click', function() {
                $('selector_dropdown').setStyle('display', 'none');
                if ($('selected_item')) {
                    $('selected_item').title = el.title;
                    $('selected_item').innerHTML = el.getElement('a').innerHTML;
                    $('selected_item').setStyle('display', 'block');
                } else {
                    var new_li = new Element ( 'li' , {
                        'id' : 'selected_item',
                        'title' : el.title,
                        'html' : el.getElement('a').innerHTML 
                    }) ;
                    new_li.injectAfter($('selector_dropdown'));
                }
                if ($('quicklinks_button_v2'))
                {
                    $('quicklinks_button_v2').href = el.title;
                }
                collapse('dropdown_contact_hq_v2', '');
            });
        });
        $$('div#dropdown_contact_hq_v2 ul li')[($$('div#dropdown_contact_hq_v2 ul li').length - 1)].getElement('a').setAttribute('id', 'last_ql_links');

        $$('a#last_ql_links').addEvent('focus', function(el) {
            //alert(document.getElementById('dropdown_contact_hq_v2').style.top);
            if (document.getElementById('dropdown_contact_hq_v2').style.top == '-5000px')
            {
                expand('droptitle_contact_hq_v2', 'dropdown_contact_hq_v2', 2, 2, 2, 2, '', 'vAuto', 'vAuto', '', '');
            }
        });

        if ($('quicklinks_button_v2'))
        {
            $('quicklinks_button_v2').addEvent('click', function(el) {
                if ($('selected_item').title) {
                    $('selector_dropdown').setStyle('display', 'block');
                    $('selected_item').setStyle('display', 'none');
                }
            });
        }

    }



    if ($('dropdown_policy_areas')) {

        $('dropdown_policy_areas').setStyle('position', 'absolute');
        $('dropdown_policy_areas').setStyle('top', '-5000px');
        //$('areas_button').setStyle('display','inline');

        $('droptitle_policy_areas').addEvent('mouseout', function(el) {
            collapse('dropdown_policy_areas', '');
            $('dropdown_policy_areas').setStyle('display', 'none');
        });
        $('droptitle_policy_areas').addEvent('click', function(el) {
            $('dropdown_policy_areas').setStyle('display', 'block');
            //expand_eu_portal('droptitle_policy_areas','dropdown_policy_areas',5,1,5,1,'','vAuto','vBottom',140);
            //expand('droptitle_policy_areas','dropdown_policy_areas',5,1,5,1,'','vAuto','vBottom',140,1610);
            //expand('droptitle_policy_areas','dropdown_policy_areas',7,1,5,1,'','vAuto','vAuto','','1717');
            if (navigator.appName == 'Microsoft Internet Explorer' && navigator.appVersion.indexOf('MSIE 7.0'))
            {
                expand('droptitle_policy_areas', 'dropdown_policy_areas', 7, 1, 5, 1, '', 'vAuto', 'vAuto', '', '435');
            } else
            {
                expand('droptitle_policy_areas', 'dropdown_policy_areas', 7, 1, 5, 1, '', 'vAuto', 'vAuto', '', '1717');
            }


        });

        $('dropdown_policy_areas').addEvent('mouseout', function(el) {
            collapse('dropdown_policy_areas', '');
            $('dropdown_policy_areas').setStyle('display', 'none');
        });
        $('dropdown_policy_areas').addEvent('mouseover', function(el) {
            $('dropdown_policy_areas').setStyle('display', 'block');
            //expand_eu_portal('droptitle_policy_areas','dropdown_policy_areas',5,1,5,1,'','vAuto','vBottom',140);		
            //expand('droptitle_policy_areas','dropdown_policy_areas',5,1,5,1,'','vAuto','vBottom',140,1610);							
            //expand('droptitle_policy_areas','dropdown_policy_areas',7,1,5,1,'','vAuto','vAuto','','1717');
            if (navigator.appName == 'Microsoft Internet Explorer' && navigator.appVersion.indexOf('MSIE 7.0'))
            {
                expand('droptitle_policy_areas', 'dropdown_policy_areas', 7, 1, 5, 1, '', 'vAuto', 'vAuto', '', '435');
            } else
            {
                expand('droptitle_policy_areas', 'dropdown_policy_areas', 7, 1, 5, 1, '', 'vAuto', 'vAuto', '', '1717');
            }

        });

        $$('div#dropdown_policy_areas ul li').each(function(el) {
            el.title = el.getElement('a').href;
            el.innerHTML = '<a href="' + el.title + '">' + el.getElement('a').innerHTML;
            + '</a>';
            el.getElement('a').addEvent('click', function() {
                $('selector_dropdown').setStyle('display', 'none');
                //$('selector_dropdown').title = el.title;
                //$('selector_dropdown').innerHTML = el.getElement('a').innerHTML;
                if ($('selected_item')) {
                    $('selected_item').title = el.title;
                    $('selected_item').innerHTML = el.getElement('a').innerHTML;
                    $('selected_item').setStyle('display', 'block');
                } else {
                    var new_li = new Element ( 'li' , {
                        'id' : 'selected_item',
                        'title' : el.title,
                        'html' : el.getElement('a').innerHTML 
                    }) ;
                    new_li.injectAfter($('selector_dropdown'));
                }
                collapse('dropdown_policy_areas', '');
            });
        });


        /*$('areas_button').addEvent('click', function(el){
        					if ($('selected_item').title) {
        					
        						location.href=$('selected_item').title;
        						$('selector_dropdown').setStyle('display','block');
        						$('selected_item').setStyle('display','none');
        					}
        				});
        				$('areas_button').addEvent('click', function(el){
        					if ($('selected_item').title) {
        					
        						location.href=$('selected_item').title;
        						$('selector_dropdown').setStyle('display','block');
        						$('selected_item').setStyle('display','none');
        					}
        				});*/

    }



    if ($('dropdown_policies')) {
        //$('dropdown_policies').setStyle('display','inline');
        buildTagSelect('dropdown_policies', 'areas_button', 'policyareas');

        $('dropdown_policies').addEvent('change', function(el) {
            window.location.href = this.value;
        });
    }
    if ($('dropdown_archives')) {
        //$('dropdown_archives').setStyle('display','inline');
        buildTagSelect('dropdown_archives', 'areas_button', 'archives');
    }

    /*
    			if ($('dropdown_archives')) {
    				
    				$('dropdown_archives').setStyle('position','absolute');
    				$('dropdown_archives').setStyle('top','-5000px');
    				$('archives_button').setStyle('display','inline');
    				
    				$('droptitle_archives').addEvent('mouseout', function(el){
    					collapse('dropdown_archives','');	
    					$('dropdown_archives').setStyle('display','none');
    				});	
    				$('droptitle_archives').addEvent('click', function(el){
    					$('dropdown_archives').setStyle('display','block');
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vBottom',140);							
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vAuto');		
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vBottom',140,1455);					
    					expand('droptitle_archives','dropdown_archives',3,2,5,1,'','vAuto','vAuto','',1455);					
    				});	
    				
    				$('dropdown_archives').addEvent('mouseout', function(el){
    					collapse('dropdown_archives','');						
    					$('dropdown_archives').setStyle('display','none');
    				});	
    				$('dropdown_archives').addEvent('mouseover', function(el){
    					$('dropdown_archives').setStyle('display','block');
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vBottom',140);							
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vAuto');	
    					//expand('droptitle_archives','dropdown_archives',5,1,5,1,'','vAuto','vBottom',140,1455);					
    					expand('droptitle_archives','dropdown_archives',3,2,5,1,'','vAuto','vAuto','',1455);					
    				});																

    				$$('div#dropdown_archives ul li').each(function(el){
    					el.title = el.getElement('a').href;
    					el.innerHTML = '<a href="javascript:void(0);">'+el.getElement('a').innerHTML;+'</a>';
    					el.getElement('a').addEvent('click', function(){
    						$('selector_dropdown').setStyle('display','none');
    						//$('selector_dropdown').title = el.title;
    						//$('selector_dropdown').innerHTML = el.getElement('a').innerHTML;
    						if($('selected_item')) {
    							$('selected_item').title = el.title;
    							$('selected_item').innerHTML = el.getElement('a').innerHTML;
    							$('selected_item').setStyle('display','block');
    						} else {
    							var new_li = new Element ( 'li' , { 'id' : 'selected_item', 'title' : el.title, 'html' : el.getElement('a').innerHTML }) ;
    							new_li.injectAfter($('selector_dropdown'));
    						}
    						collapse('dropdown_archives','');
    					});
    				});
    				
    				$('archives_button').addEvent('click', function(el){
    					if ($('selected_item').title) {
    					//if ($('selector_dropdown').title) {
    						//alert($('selector_dropdown').innerHTML);
    						//window.open($('selector_dropdown').title);
    						//location.href=$('selected_item').title;
    						
    						//if ($('whatwasnew')) {
    						/*
    						var myHTMLRequest = new Request.HTML(
    						
    							onSuccess(responseTree, responseElements, responseHTML, responseJavaScript)
    						
    						).get($('selected_item').title);
    						*/
    //alert(myHTMLRequest.innerHTML);
    //}
    //test = $('selected_item').title;
    //alert(test);

    //if ($('mediabox_general_content')) {
    //alert(test);
    //}
    /*
    					request = new Request.HTML({
    						url: $('selected_item').title,
    						onSuccess: function(responseTree, responseElements, responseHTML, responseJavaScript){
    							responseElements.each(function(element){
    								if(element.get('id') == 'mediabox_general_content'){
    									$('mediabox_general_content').adopt(element);
    								} else {
    									alert('hello');
    								}
    							});
    						}
    					}); */
    /*
    						var thisIdEl		= 'general';
    						var thisIdElement	= 'mediabox_'+thisIdEl+'_content';
    						//alert(thisIdElement);
    					
    						var myHTMLRequest = new Request.HTML({
    							onSuccess: function(responseTree, responseElements, responseHTML, responseJavaScript){
    								
    								responseElements.each(function(element){
    									
    									if(element.get('id') == thisIdElement){

    										element.setStyle('display','none');
    										element.replaces($(thisIdElement));
    										element.getElements('h5').each(function(el){
    											el.addClass('toggler');
    											el.innerHTML = '&nbsp;'+el.innerHTML;
    											var imgPlus = new Element('img', {
    												src		: '/wel/images/eu_portal/plus.gif',
    												width	: '9',
    												height 	: '9',
    												alt		: ''
    											});
    											imgPlus.injectTop(el);
    										});
    										
    										element.getElements('div.elaccordion').each(function(el){
    											el.addClass('element');
    										});
    									}
    									
    								});
    								
    								var tabRunning 	= Cookie.read(cookieTabRunning);
    								var tabInList	= isTabInList(tabRunning);
    								if ( (!tabRunning) || (!tabInList) ) {
    									$$('ul#mediabox_tabs li:first-child a').each(function(el) {			   
    										tabRunning = el.id;
    									});
    								}
    								//alert(tabRunning);
    								
    								$$('#'+tabRunning).each(function(el) {
    									if ($('tabContainer').hasChild('tabMenu')){
    										initAccordionHome(el);
    										//alert('pan  -- '+el);
    									} else {
    										initAccordion(el);
    										//alert('pouf  -- '+el);
    									}
    								});
    								
    							}
    						}).get($('selected_item').title);
    						
    						$('selector_dropdown').setStyle('display','block');
    						$('selected_item').setStyle('display','none');
    					
    					}
    				});
    				//$('dropdown_contact_hq_v2').setStyle('display','block');				
    				
    			}
    */
    /*
    				$('archives_button').addEvent('click', function(el){
    					if ($('selected_item').title) {*/
    //if ($('selector_dropdown').title) {
    //alert($('selector_dropdown').innerHTML);
    //window.open($('selector_dropdown').title);
    /*location.href=$('selected_item').title;
    						$('selector_dropdown').setStyle('display','block');
    						$('selected_item').setStyle('display','none');
    					}
    				});*/


    //	Read Parser
    /*
    		if ($('whatwasnew')) {
    			var req = new Request.HTML({url:demo_path+'data.html', 
    			onSuccess: function(html) {
    				//Clear the text currently inside the results div.
    				$('result').set('text', '');
    				//Inject the new DOM elements into the results div.
    				$('result').adopt(html);
    			},
    			//Our request will most likely succeed, but just in case, we'll add an
    			//onFailure method which will let the user know what happened.
    			onFailure: function() {
    				$('result').set('text', 'The request failed.');
    			}
    	});
    		}*/

    //	Tabs Generation

    //$$('#tabContainer .mediastabs li').each(function(el,index){
    $$('#tabContainer .mediastabs li').each(function(el, index) {

        href_link = el.getElement('a').href;
        pos_char_anc = el.getElement('a').href.lastIndexOf('#') + 1;
        id_link = href_link.substring(pos_char_anc, href_link.length);
        text_link = el.getElement('a').innerHTML;

        class_attribut = '';
        if (index == 0) {
            class_attribut = " class = active";
        }
        new_inner_li = '<a id="' + id_link
        + '" href="mediabox_' + id_link
        + '_content"' + class_attribut
        + '><span>' + text_link
        + '</span></a>';
        el.innerHTML = new_inner_li;
    });

    $$('#tabContainer .tabContent').each(function(el) {
        $$('h2.tabsTitel').dispose();
        $$('a.tabsAnchor').dispose();
        $$('.tabMenu').addClass('tabMenu_js');
        el.addClass('tabcontent_js');
    });

    /* ------------------------ */



    $$('#tabContainer .tabContent div h5').each(function(el) {
        if (!(($chk($('whatwasnew'))) && ($chk($('mediabox_general_content'))))) {
            el.addClass('toggler');
            el.innerHTML = '&nbsp;' + el.innerHTML;
            if ($('whatwasnew') || $('sitemapNav')) {
                var suffix = '-operator';
            } else {
                var suffix = '';
            }



            var imgPlus = new Element('img', {
                src : '/wel/images/eu_portal/plus' + suffix + '.gif',
                width : '9',
                height : '9',
                alt : ''
            });

            imgPlus.injectTop(el);
        }
    });
    $$('div.elaccordion').each(function(el) {
        el.addClass('element');
    });

    // Search field clear
    if ($(idSearchField))
    {
        $(idSearchField).addEvent('focus', function(el) {
            this.set('title', this.get('value'));
            this.set('value', '');
        });

        $(idSearchField).addEvent('blur', function(el) {
            if (this.get('value') == '') {
                this.set('value', this.get('title'));
            }
        });
    }

    //checkBackBoxMootools();	

    $$('.hiddenJSExec').each(function(el) {
        el.removeClass('hiddenJSExec');
    });







    if ($$('div.eu_portal_audiovisual_content_container').length > 0)
    {
        nb_li = $$('div.eu_portal_audiovisual_content_container > ul.eu_portal_audiovisual_content > li').length;
        current_video_position = 1;

        $$('div.eu_portal_audiovisual_content_container > ul.eu_portal_audiovisual_content > li').each(function(el, index) {

            if (index > 0)
            {
                el.addClass('displaynone');
            } else
            {
                el.addClass('current');
            }

        });

        if (nb_li > 1)
        {
            media_pagin = '<ul class="mediaPagin"><li class="gotofirst"><a title="First item" href="javascript:void(0)" class="navIco navFirstDisabled"><img border="0" alt="First item" src="/wel/template-2009/images/none.gif"/><span class="noCss">First item</span></a></li><li class="gotoprevious"><a title="Previous item" href="javascript:void(0)" class="navIco navPreviousDisabled"><img border="0" alt="Previous item" src="/wel/template-2009/images/none.gif"/><span class="noCss">Previous item</span></a></li><li class="mediaNumber">1/' + nb_li + '</li><li class="gotonext"><a title="Next item" href="javascript:void(0)" class="navIco navNext"><img border="0" alt="Next item" src="/wel/template-2009/images/none.gif"/><span class="noCss">Next item</span></a></li><li class="gotolast"><a title="Last item" href="javascript:void(0)" class="navIco navEnd"><img border="0" alt="Last item" src="/wel/template-2009/images/none.gif"/><span class="noCss">Last item</span></a></li></ul>';

            $$('div.media_pagin_container').set('html', media_pagin);
        } else
        {
            $$('div.download_link_container').setStyle('bottom', '-29px');
        }



        $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addEvent('click', function(el) {

            current = $$('ul.eu_portal_audiovisual_content > li.current');

            current.addClass('displaynone');
            current.removeClass('current');

            $$('ul.eu_portal_audiovisual_content > li')[0].removeClass('displaynone');
            $$('ul.eu_portal_audiovisual_content > li')[0].addClass('current');

            current_video_position = 1;
            $$('div.media_pagin_container > ul.mediaPagin > li.mediaNumber').set('html', current_video_position + '/' + nb_li);

            $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEndDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEnd');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNextDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNext');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirstDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirst');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPreviousDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPrevious');
        });

        $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addEvent('click', function(el) {

            if (current_video_position > 1)
            {
                current = $$('ul.eu_portal_audiovisual_content > li.current');

                current.getPrevious().removeClass('displaynone');
                current.getPrevious().addClass('current');

                current.addClass('displaynone');
                current.removeClass('current');

                current_video_position --;
                $$('div.media_pagin_container > ul.mediaPagin > li.mediaNumber').set('html', current_video_position + '/' + nb_li);
            }

            if (current_video_position < nb_li) 
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEndDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEnd');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNextDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNext');
            } else
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEndDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEnd');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNextDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNext');
            }

            if (current_video_position > 1)
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirstDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirst');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPreviousDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPrevious');
            } else
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirstDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirst');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPreviousDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPrevious');
            }

        });

        $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addEvent('click', function(el) {

            if (current_video_position < nb_li)
            {
                current = $$('ul.eu_portal_audiovisual_content > li.current');

                current.getNext().removeClass('displaynone');
                current.getNext().addClass('current');

                current.addClass('displaynone');
                current.removeClass('current');

                current_video_position ++;
                $$('div.media_pagin_container > ul.mediaPagin > li.mediaNumber').set('html', current_video_position + '/' + nb_li);
            }

            if (current_video_position < nb_li) 
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEndDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEnd');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNextDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNext');
            } else
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEndDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEnd');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNextDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNext');
            }

            if (current_video_position > 1)
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirstDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirst');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPreviousDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPrevious');
            } else
            {
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirstDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirst');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPreviousDisabled');
                $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPrevious');
            }
        });

        $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addEvent('click', function(el) {

            current = $$('ul.eu_portal_audiovisual_content > li.current');

            current.addClass('displaynone');
            current.removeClass('current');

            $$('ul.eu_portal_audiovisual_content > li')[nb_li - 1].removeClass('displaynone');
            $$('ul.eu_portal_audiovisual_content > li')[nb_li - 1].addClass('current');

            current_video_position = nb_li;
            $$('div.media_pagin_container > ul.mediaPagin > li.mediaNumber').set('html', current_video_position + '/' + nb_li);

            $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').addClass('navEndDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotolast > a').removeClass('navEnd');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').addClass('navNextDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotonext > a').removeClass('navNext');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').removeClass('navFirstDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotofirst > a').addClass('navFirst');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').removeClass('navPreviousDisabled');
            $$('div.media_pagin_container > ul.mediaPagin > li.gotoprevious > a').addClass('navPrevious');
        });

    }





});


function changeUrl() {

    var dropdown = document.getElementById('select_selector_dropdown');
    if (dropdown.options[dropdown.selectedIndex].value != - 1)
    {
        document.location.href = dropdown.options[dropdown.selectedIndex].value;
    }
}


