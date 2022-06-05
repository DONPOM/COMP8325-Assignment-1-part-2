
$j(document).ready(function(){
	// promo + footer tables
	var coolTables=["2","3"];
	$j('.promobox_list_block').each(function(){
		$j(this).addClass('plb plb'+$j(this).find('tr:nth-child(1) td').length);
		for (var i = 0; i < coolTables.length; i++) {
			var pr_t = 0;
			var p_cap = coolTables[i];
			$j(this).after('<table cellspacing="0" cellpadding="0" class="promobox_list_block plb plb'+p_cap+' feed_t"></table>');
			$j(this).find('td').each(function() {
				if (pr_t == 0) {
					$j('.feed_t').append('<tr class="feed_r"></tr>');
				}
				//
				$j(this).clone().appendTo('.feed_r');
				//
				pr_t = pr_t+1;
				if (pr_t == p_cap) {
					pr_t = 0;
					$j('.feed_r').removeClass('feed_r');
				}
			});
			//
			var desc = 0;
			if (pr_t) {
				desc = p_cap-pr_t;
			}
			//
			if (desc && pr_t) {
				for (var l = 0; l < desc; l++) {
					$j('.feed_r').append('<td class="i-hold-weight"></td>');
				}
			}
			//
			$j('.feed_t').removeClass('feed_t');
			$j('.feed_r').removeClass('feed_r');
		}
		$j('.feed_r').removeClass('feed_r');
	});
	//
});
