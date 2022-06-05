
function loadUserForms(){
	new Ajax.Request(ajax, {
		parameters: {
			pos: 'getuserforms',
			cookie: document.cookie
		},
		onSuccess:
			function(t){
				var forms = t.responseText.evalJSON();
				if( forms.size() > 0 ){
					var table = '<table width="100%" id="user-forms-table" cellspacing="0" cellpadding="0"><tr style="background: url(images/blog-settings-fade.jpg) repeat-x; height: 28px;"><th>Name</th><th style="width:140px;"></th><th>Location</th><th>Total Entries</th><th>Last Entry</th></tr>';
					var alt = 0;
					forms.each(
						function(form){
							var form_name = form.name.length < 20 ? form.name : form.name.substr(0, 19)  + '...';
							var site_name = form.site.length < 20 ? form.site : form.site.substr(0, 19)  + '...';
							table += '<tr class="form-export-row-'+alt+'"><td>'+form_name+'</td><td align="center"><a href="#" onclick="viewFormData(\''+form.ucfid+'\',\''+form.user+'\',\''+form.hash+'\'); return false;"><img src="images/view-entries.gif" alt="View"  style="margin-right:8px;"  /></a><a href="exportFormData.php?ucfid='+form.ucfid+'&user_id='+form.user+'&hash='+form.hash+'"><img src="images/export-data.gif" alt="Download" /></a></td><td><a href="'+form.link+'" target="_blank" title="'+form.site+'">'+site_name+'</a></td><td>'+form.total+'</td><td>'+form.last_submitted+'</td></tr>';
							alt = 1 - alt;
						}
					);
					table += '</table>';
					$('user-forms-data').update(table);
				}
			}
		}
	);
}

function viewFormData(ucfid, userid, hash){
	window.open('viewFormData.php?ucfid='+ucfid+'&user_id='+userid+'&hash='+hash, 'weebly_view_form', 'height=550,width=910,toolbar=yes,status=yes,resizable=yes,scrollbars=yes,dependent=yes');
}
