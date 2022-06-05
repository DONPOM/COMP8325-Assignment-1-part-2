
	function showData(models) {
		$.each(window.Plugin.Models, function(index) {
			if(index == 0) {
				var model = models[index];
				$('#chat_name').html(model.Name +' <span>'+ model.Age +'</span>')
			}		
			if(index > 0) {
				var model = models[index];
				$(".model-name:eq("+(index-1)+")").text(model.Name);
				$(".model-status:eq("+(index-1)+")").text(model.Status.toUpperCase()).addClass(model.Status.toLowerCase());	
			}	
		});
		
	}
		
		
	$(document).ready(function() {	
		$(".btn_chat_with_me").attr("href", window.Plugin.Models[0].Url);
		
		$("#filters .item").attr("href", window.Plugin.Models[0].Url);
		
		var model_ids = [];		
		$.each(window.Plugin.Models, function(i, model) {		
			model_ids.push(model.UserId);	
			if(i>0) {
				$(".og_btn_chat:eq("+(i-1)+")").attr("href", model.Url);				
			}	
		});	
		
				if(model_ids.length) {
			$.ajax({			   type : 'POST',
				   data : 'model_ids=' + model_ids + '&w=783' + '&c=DP' + '&b=AU',
				   url : '/delivery/show/hits.php',
				   success : function(data) {  
							console.log(data+'--');
					}
			});
		}
				
		$(".click_on_girl, a").click(function() {			
			StopLivePreview();
		});
		
		$(window).blur(function(){				
				setTimeout(function(){StopLivePreview()},120000);	
				$(window).focus(function(){
					StartLivePreview()
				});
		});
		

	});	