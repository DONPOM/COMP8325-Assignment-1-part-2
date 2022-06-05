$( "#loginform" ).submit(function( event ) {
  onLoginSubmit();
  event.preventDefault();
});

function onLoginSubmit() {
	var email = $('#your-email').val();
	var pw = $('#your-password').val();
//	console.log("email: " + email);
//	console.log("password: " + pw);
	
	$('#loginError').html('');	
	$('#loginButtonDiv').hide();
	$('#loginLoading').show();
	
	$.ajax(
	{
		type: 'POST',
		url: bURL + 'login',
		data: { email: email, password: pw },
		dataType: "json",
		xhrFields:{withCredentials:true},
		success: function(data) {
//			showAllProperties(data);
			if(data.success == true) {
				window.location = 'app';
			} else {
				$('#loginError').html('please try again');
				$('#loginButtonDiv').show();
				$('#loginLoading').hide();				
			}
		}
	}
	);	
}
	
function openResetDialog()
{
	$('#reset').show();
}

function onResetSubmit()
{
	var email = $('#reset-email').val();
	
	$('#resetError').html('');	
	$('#resetButtonDiv').hide();
	$('#resetLoading').show();
	
	
	$.ajax(
	{
		type: 'POST',
		url: bURL + 'reset_password',
		data: { email: email },
		dataType: "json",
		xhrFields:{withCredentials:true},
		success: function(data) {
			showAllProperties(data);
			$('#resetLoading').hide();				
			$('#resetButtonDiv').show();
			if(data.success == true) {
				$('#resetError').html('new password sent.');
			} else {
				$('#resetError').html('error');
			}
		}
	}
	);	
}

function showAllProperties(data) 
{	
	console.log("showAllProperties()");
	for (var key in data) 
	{
		console.log(key + ": " + data[key]);
	}
}