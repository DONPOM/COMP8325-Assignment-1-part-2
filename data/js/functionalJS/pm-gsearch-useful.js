
var gcsLoadCallback = function() {
	$j(document).ready(function() {
		var cse_id = '006787088747114277952:hmjq3gmjkqc';
		var searchInputs = document.getElementsByName('s');
		var formCount = 0;
		for (var i = 0; i < searchInputs.length; ++i) {
			var form = searchInputs[i].form;
			if (!form.action.match(/searchx/))
				continue;
			if (!form.id.length) {
				form.id = 'searchForm'+formCount;
				formCount++;
			}
			google.search.CustomSearchControl.attachAutoCompletion(
				cse_id,
				searchInputs[i],
				form.id
			);
		}
	});
};
IncludeJavaScript("http://www.google.com/jsapi?autoload=%7B%22modules%22%3A%5B%7B%22name%22%3A%22search%22%2C%22version%22%3A1%2C%22callback%22%3A%22gcsLoadCallback%22%7D%5D%7D");
