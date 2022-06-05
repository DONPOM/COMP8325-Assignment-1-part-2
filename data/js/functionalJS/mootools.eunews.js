
window.addEvent('domready', function() {

    /*gets the language of the document based on the "lang" attribute of the HTML tag*/
    var lang = $$('html').get('lang');
    //alert (lang);

    var eu_news_url = '/newsroom/eunews/index_' + lang + '.htm?ea-ignore=true';

    var eu_news_more_link = $('mediabox_news_content').getElement('ul.morenews');

    var icons = '<a title="Deutsch (de)" href="http://europa.eu/newsroom/index_de.htm"><img src="/wel/images/languages/lang_de.gif" class="alIco" border="0" alt="Deutsch (de)"></a><a title="English (en)" href="http://europa.eu/news/index_en.htm"><img src="/wel/images/languages/lang_en.gif" class="alIco" border="0" alt="English (en)"></a><a title="Fran&ccedil;ais (fr)" href="http://europa.eu/news/index_fr.htm"><img src="/wel/images/languages/lang_fr.gif" class="alIco" border="0" alt="Fran&ccedil;ais (fr)"></a>';

    //var span='';


    /*sets default to English on language fallback*/
    /*if(lang != 'en' && lang != 'fr' && lang != 'de'){
    		eu_news_url = '/newsroom/index_en.txt';
    	}*/



    var XMLreq = new Request.HTML({
        url: eu_news_url,
        method: 'get',
        noCache: 'true',
        onSuccess: function(html, xml) {
            /*	
            			$('ajaxload').set('text','');
            				$('ajaxload').adopt(xml.getElements('p'));
            			
            			if(lang != 'en' && lang != 'fr' && lang != 'de'){
            				$('ajaxload').getElements('strong').each(function(el){
            					
            					this.title = el.getElement('a').get('title');
            					this.span = new Element('span', {'class': 'nolink ajax', 'html' : this.title+icons});
            					el.set('text','');
            	  			el.adopt(this.span);
            				});				
            			}*/

            $('ajaxload').set('text', '');
            /*if(lang != 'en' && lang != 'fr' && lang != 'de'){
                      $('ajaxload').adopt(xml.getElements('div.icons'));
                    }
                    else
                    {
                      $('ajaxload').adopt(xml.getElements('div.noicons'));
                    }*/
            $('ajaxload').adopt(xml.getElements('div'));
        },

        onFailure: function() {

            $('ajaxload').set('text', '<p>could not load news</p>');

        },
        onRequest: function() {

            var par = new Element('p', {
                'class': 'first',
                'align': 'center'
            });
            var img = new Element('img', {
                'src': '/wel/images/eu_portal/ajax-loader.gif'
            });
            $('ajaxload').set('text', '');
            $('ajaxload').adopt(par);
            par.adopt(img);
        }
    });


    XMLreq.send();


});
