
function setupDesigner() {
	new Ajax.Request(ajax, {
		parameters : Object.extend({
			pos: 'setupdeveloper',
			cookie: document.cookie
		}, $('designer-signup-form').serialize(true)),
		onSuccess : function(t) {
			if(t.responseText.match(/%%SUCCESS%%/)) {
				window.location = "userHome.php?page=developer&subpage=clients";
			} else {
				$('setup-designer-error').update(t.responseText).show();
			}
		}
	})
}
