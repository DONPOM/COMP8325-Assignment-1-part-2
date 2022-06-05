
$(function() {

var api;

api = $('#lcol-tabs > .tabs').tabs($('#lcol-tabs > .panes > .pane'), {
    current: 'active',
    initialIndex: 0,
    onBeforeClick: function(e, i) {
        $('#lcol-tabs > .tabs').removeClass('state' + (this.getIndex() + 1)).addClass('state' + (i + 1));
        $('#lcol-tabs > .panes > .tab_title').removeClass('active').eq(i).addClass('active');
    },
	onClick: function() {
		$(document).trigger("ui_updated");
	}
}).data('tabConfig', { 'initialIndex': 0 }).data('tabs');


});
