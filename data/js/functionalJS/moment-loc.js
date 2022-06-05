var momentLangMap = {
	'de-DE': 'de',
	'en-GB': 'en-gb',
	'en-US': 'en',
	'es-ES': 'es',
	'es-MX': 'es',
	'es-US': 'es',
	'fr-CA': 'fr-ca',
	'fr-FR': 'fr',
	'it-IT': 'it',
	'nb-NO': 'nb',
	'nl-NL': 'nl',
	'sv-SE': 'sv',
	'zz-ZZ': 'de'
}

var momentLang = (webs && webs.locale && momentLangMap[webs.locale]) ? momentLangMap[webs.locale] : 'en';

var momentPath = (momentLang === 'en') ? 'moment' : 'moment-langs/' + momentLang;

define([momentPath
], function(momentjs) {
	return require('moment');
});