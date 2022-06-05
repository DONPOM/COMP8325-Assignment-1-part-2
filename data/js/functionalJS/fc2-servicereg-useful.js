
$('#sevice_regist_table tbody').sortable({
	items: 'tr',
	cancel: 'tr td a',
	update: function(ev, ui){
		var order = [];
		$('#sevice_regist_table tbody tr').each(function(k,v){
			order.push(v.title);
		});
		$.ajax({
			"type": 'POST',
			"url": "/update_service_list_order.php",
			"data": 'order=' + encodeURIComponent(order.join(','))
		});
		even_line();
	}
});
$('#sevice_regist_table').disableSelection();
$('#sevice_regist_table tr').each(function(i, tr){
	$('td', tr).each(function(j, td){
		var td=$(td);
		td.css({width: td.width()});
	});
});

function even_line() {
	$('#sevice_regist_table tbody tr').each(function(i, tr){
		if(i % 2 == 1){
			$(tr).addClass('even_line');
		}else{
			$(tr).removeClass('even_line');
		}
	});
}
even_line();