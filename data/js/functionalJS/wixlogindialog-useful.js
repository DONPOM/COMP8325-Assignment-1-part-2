
        $(document).ready(function ()
        {
            LoginDialog.init({
                "locale": lng,
                "flow": "default",
                "userApi": userApi,
                "postLogin": function (response)
                {
                    document.location.href = my_account_url;
                },
                "postSignUp": function (response)
                {
                    document.location.href = postreg_url;
                },"onLoad": function ()
                {

                },
                userApiOptions: {
                    "corsEnabled": false,
                    "usersDomain": "https://sslusers.wix.com/wix-users"
                }
            });
        });
    