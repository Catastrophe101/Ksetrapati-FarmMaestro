from pickle import GET

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/index, methods=['GET', 'POST']')
def index():
	result = None
	if request.method == 'POST':
		moisture = float(request.form['number'])
		nitrogen = float(request.form['N'])
		phosphorus = float(request.form['P'])
		potassium = float(request.form['K'])
		result='No result'
	return render_template('index.jinja2', result=result)


if __name__ == '__main__':
    app.debug = True
    app.run()