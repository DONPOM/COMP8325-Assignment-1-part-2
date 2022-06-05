
	//array of language supported by application
	var langArray=new Array("es","fr","de","it");
	//default langauge
	//var lang='en';
	
	var navigatorLanguage = (navigator.language) ? navigator.language: navigator.browserLanguage;
	
	var languageVar = navigatorLanguage.substr(0,2).toLowerCase();
	
	var subLanguageVar = navigatorLanguage.substr(3,2).toLowerCase();

	if(languageVar!='en')
	{
		var jsFile = null;
		//check for language
		switch (languageVar) {
		case "zh":
			jsFile="http://w.sharethis.com/share4x/js/local/cn/message.js";
			if ("tw" == subLanguageVar) {
				jsFile="http://w.sharethis.com/share4x/js/local/tw/message.js";
			}
			break;
		case "es":
			jsFile="http://w.sharethis.com/share4x/js/local/es/message.4619903adb026e438622c3b385adbca4.js";
			break;
		case "fr":
			jsFile="http://w.sharethis.com/share4x/js/local/fr/message.1d06ed11af247ad5da457345ed755c71.js";
			break;					
		case "de":
			jsFile="http://w.sharethis.com/share4x/js/local/de/message.a915285c7f2e99521677f6d9e7234917.js";
			break;
		case "it":
			jsFile="http://w.sharethis.com/share4x/js/local/it/message.1b86943add920b4af249f44eaf7ecf95.js";
			break;
		case "ja":
			jsFile="http://w.sharethis.com/share4x/js/local/jp/message.js";
			break;	
		case "nl":
			jsFile="http://w.sharethis.com/share4x/js/local/nl/message.js";
			break;	
		case "pt":
			jsFile="http://w.sharethis.com/share4x/js/local/pt/message.js";
			break;
		case "ru":
			jsFile="http://w.sharethis.com/share4x/js/local/ru/message.js";
			break;
		case "tr":
			jsFile="http://w.sharethis.com/share4x/js/local/tr/message.js";
			break;	
		case "ua":
			jsFile="http://w.sharethis.com/share4x/js/local/ua/message.js";
			break;	
		case "vn":
			jsFile="http://w.sharethis.com/share4x/js/local/vn/message.js";
			break;	
		default: 
		}
		//message file to include
		if (jsFile != null) {
			document.write('<script type="text/javascript" src="'+ jsFile +'"><\/script>');
		}
	}
