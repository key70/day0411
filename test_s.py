from flask import Flask,render_template,request
from day0411 import func
app = Flask(__name__)
import day0411.func




@app.route('/test.do',methods=['GET','POST'])
def test():

    domain = day0411.func.getDomain()


    # age = ''
    # workclass = ''
    # education = ''
    # occupation = ''
    # race = ''
    # sex = ''
    # hoursperweek = ''
    # data = ''
    msg = ''
    if request.method == 'POST':
        age = request.form['age']
        workclass = request.form['workclass']
        education = request.form['education']
        occupation = request.form['occupation']
        race = request.form['race']
        sex = request.form['sex']
        hoursperweek = request.form['hoursperweek']




        data = func.adult_d(age,workclass,education,occupation,race,sex,hoursperweek)

        if data[0] == 1:
            msg = "대출 가능"
        else:
            msg = '대출 불가능'
    return render_template('test.html',msg=msg, domain=domain)


if __name__=='__main__':
    app.run(debug=True)