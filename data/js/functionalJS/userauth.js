/**
 *
 * Handles user authentication.
 * Defines Auth functions: login, register and getToken.
 *
 * Compatible with require and legacy NYTD
 * @module Auth
 * @class User
 * @static
 *
 * @this Auth
 * @version $Id$
 *
 */

/**
 *
 * Usage:
 *
 *    Place a form on the page. Call the register and login functions as follows. There is no need to get token,
 *    and no need for form fields other than the ones below:
 *
 *    Auth.register({
 *          email: {String},           // User email to register
 *          password: {String},        // Password
 *          confPassword: {String},    // Confirmation password
 *          specialOffers: {Boolean},  // True if user wants to subscribe to newsletter
 *    })
 *
 *    Auth.login({
 *          user: {String},            // User name or email
 *          password: {String},        // Password
 *          remember: {Boolean}        // True if we want login cookie to survive session
 *   })
 *
 *   If you need to have access to the token for other apis, use the following assynchronous call:
 *
 *   Auth.getToken().done(<do_something>).fail(<do_something_else)
 *
 *   Done promise will be called with parameters (token_value, token_header_name, cached)
 *   Fail promise will be called with parameters (error_name)
 *
 **/

(
    function(root, factory) {
        'use strict';

        if (typeof exports !== 'undefined') {
            exports.Auth = factory(null, $);
        }
        else if (typeof define === "function" && define.amd) {
            define('foundation/lib/auth/userauth',['foundation/hosts', 'jquery/nyt'], function(hosts, $) {
                return factory(hosts, $);
            });
        }
        else if (typeof root.NYTD === 'object') {
            var NYTD = root.NYTD;
            var Auth = factory(NYTD.Hosts, NYTD.jQuery);
            NYTD.Auth = NYTD.Auth || {};
            NYTD.Auth.register = Auth.register;
            NYTD.Auth.login = Auth.login;
            NYTD.Auth.getToken = Auth.getToken;
        }
    } (this, function(hosts, $) {
        'use strict';

        var token;
        var tokenExpiration;
        var isRequestingToken;
        var isTokenApiDead;
        var tokenDeferred;

        var WWW_HOST = hosts ? (hosts.www ? hosts.www : hosts.wwwHost) : '//www.nytimes.com';
        var MYACCOUNT_HOST = hosts ? (hosts.myaccount ? hosts.myaccount : hosts.myaccountHost) : 'https://myaccount.nytimes.com';

        var TOKEN_SVC = WWW_HOST + '/svc/profile/token.jsonp?callback=?';
        var TOKEN_TIMEOUT = 5000;
        var TOKEN_EXPIRATION = 600000;
        var TOKEN_HEADER_NAME = 'X-Nyt-Sartre-Token';

        var REGI_PAGE = MYACCOUNT_HOST + "/register";
        var LOGIN_PAGE = MYACCOUNT_HOST + "/auth/login";

        var ERROR_TIMEOUT = 'timeout';
        var ERROR_CORRUPT = 'corrupt';
        var ERROR_DEAD = 'dead';

        /**
         * Register a user
         *
         * @method register
         * @public
         *
         * @param options {Object} Form options
         *     @param options.email {String} Email we want to register
         *     @param options.password {String} Password
         *     @param options.confPassword {String} Password confirmation
         *     @param options.specialOffers {Boolean} True if the user wants to subscribe to newsletter
         */
        var register = function(options) {

            var regiPage = REGI_PAGE + '?' + getAuthQueryString();

            var postForm = function() {

                // creates form
                var form = ['<form action="', regiPage, '" method="POST" class="hidden" id="authRegiForm">',
                                '<input name="is_continue" value="1" type="text">',
                                '<input name="email_address" value="', options.email, '" type="text">',
                                '<input name="password1" type="password">',
                                '<input name="password2" type="password">',
                                options.specialOffers ? '<input name="subscribe[]" type="text" value="MM">' : '',
                                '<input name="token" value="', token, '" type="text">',
                                '<input name="expires" value="', tokenExpiration, '">',
                            '</form>'].join('');
                $(document.body).append(form);

                // updates password values without writing them to the page
                $('#authRegiForm input[name=password1]').val(options.password);
                $('#authRegiForm input[name=password2]').val(options.confPassword);

                // submit
                $('#authRegiForm').submit();
            };

            var redirect = function() {
                window.location.replace(regiPage);
            };

            // make request to token
            // then post response to regi
            getToken().done(postForm).fail(redirect);
        };

        /**
         * Logs in a user
         *
         * @method login
         * @public
         *
         * @param options {Object} Form options
         *     @param options.user {String} username or email of the user
         *     @param options.password {String} Password
         *     @param options.remember {Boolean} True if we want cookie to survive session
         */
        var login = function(options) {

            // creates login page url with return uri
            var loginPage = LOGIN_PAGE + '?' + getAuthQueryString();

            var postForm = function() {

                // creates form
                var form = ['<form action="', loginPage, '" method="POST" class="hidden" id="authLoginForm">',
                                '<input name="is_continue" value="1" type="text">',
                                '<input name="userid" value="', options.user, '" type="text">',
                                '<input name="password" type="password">',
                                '<input name="remember" value="', options.remember, '" type="text">',
                                '<input name="token" value="', token, '" type="text">',
                                '<input name="expires" value="', tokenExpiration, '">',
                            '</form>'].join('');
                $(document.body).append(form);

                // updates password values without writing them to the page
                $('#authLoginForm input[name=password]').val(options.password);

                // submit
                $('#authLoginForm').submit();
            };

            var redirect = function() {
                window.location.replace(loginPage);
            };

            // make request to token
            // then post response to regi
            getToken().done(postForm).fail(redirect);
        };

        /**
         * Makes a token request
         *
         * @method getToken
         * @public
         */
        var getToken = function() {

            // if there already is a token request on the way,
            // don't make another. Also reuse the same promise, since it relates to
            // the same request
            if (isRequestingToken) {
                return tokenDeferred.promise();
            }

            // deferred states are resolved; start a new request
            tokenDeferred = new $.Deferred();

            // if we already have a valid token,
            // return it instead of making another call
            if (isTokenValid()) {
                tokenDeferred.resolve(token, TOKEN_HEADER_NAME, true);
                return tokenDeferred.promise();
            }

            // if the token api is having trouble,
            // stop calling it for a while
            if (isTokenApiDead) {
                tokenDeferred.reject(ERROR_DEAD);
                return tokenDeferred.promise();
            }

            // and we start to request for a token
            isRequestingToken = true;

            // set up timeouts for jsonp request
            var tokenTimer = setTimeout(function() {
                isRequestingToken = false;
                setTokenApiDead();
                tokenDeferred.reject(ERROR_TIMEOUT);
            }, TOKEN_TIMEOUT);

            $.getJSON(TOKEN_SVC, null, function(resp) {
                isRequestingToken = false;
                if (resp && resp.data && resp.data.token) {
                    token = resp.data.token;
                    clearTimeout(tokenTimer);
                    updateTokenExpiration();
                    tokenDeferred.resolve(token, TOKEN_HEADER_NAME, false);
                } else {
                    // token api failed; set token dead for a few minutes
                    // so we don't hammer the api with requests
                    setTokenApiDead();
                    clearTimeout(tokenTimer);
                    tokenDeferred.reject(ERROR_CORRUPT);
                }
            });

            return tokenDeferred.promise();
        };

        /**
         * Disables token api requests for 5 minutes
         *
         * @method setTokenApiDead
         * @private
         */
        var setTokenApiDead = function() {
            isTokenApiDead = true;
            setTimeout(setTokenApiAlive, 300000 /* 5 minutes */);
        };

        /**
         * Enables requests for token api
         *
         * @method setTokenApiDead
         * @private
         */
        var setTokenApiAlive = function() {
            isTokenApiDead = false;
        };

        /**
         * Resets token expiration
         *
         * @method updateTokenExpiration
         * @private
         **/
        var updateTokenExpiration = function() {
            tokenExpiration = (new Date()).getTime() + TOKEN_EXPIRATION;
        };

        /**
         * Verifies if token is still valid, ie, tokenExpiration is in the future
         *
         * @method isTokenValid
         * @private
         * @return {Boolean}
         */
        var isTokenValid = function() {
            return tokenExpiration > (new Date()).getTime();
        };

        /**
         * Creates a query string for the login and register applications to redirect
         *  back to the original page
         *
         *  @method getAuthQueryString
         *  @private
         *  @return {String}
         */
        var getAuthQueryString = function () {
            var query;
            var locObj = window.location;
            var paramObj = {
                URI: locObj.protocol + '//' + locObj.hostname + '/' + locObj.pathname.replace(/^\//, '')
            };

            if (locObj.search.length > 1) {
                query = locObj.search.substr(1);
                query = query.replace(/(\?|&)?gwh=([^&]+)/g, '');
                query = query.replace(/(\?|&)?gwt=([^&]+)/g, '');
                query = query.replace(/%/g, 'Q');
                paramObj.OQ = query;
            }

            return $.param(paramObj);
        };

        // defines public function object
        return {
            register: register,
            login: login,
            getToken: getToken
        };
    })
);

