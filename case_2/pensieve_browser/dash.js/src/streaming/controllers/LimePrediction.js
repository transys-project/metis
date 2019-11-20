var LimePrediction = function () {
    let models = [];
    let centers = [];
    let model = {};

    for (let i = 0; i < 10; i++) {
        model['coef_'] = tf.randomNormal([25, 6]);
        model['intercept_'] = tf.randomNormal([1, 6]);

        models.push(model);
        centers.push(tf.randomNormal([1, 25]));
    }


    this.select_cluster = function (input, centers) {
        let min_distance = 1e10;
        let min_index = 0;
        for (let i = 0; i < 10; i++) {
            let dis = tf.sum(tf.square(tf.sub(input, centers[i]))).dataSync()[0];
            if (dis < min_distance) {
                min_distance = dis;
                min_index = i;
            }
        }
        return min_index;
    }

    this.predict = function (inputTensor) {
        let selected_index = this.select_cluster(inputTensor, centers);
        let result_action = tf.argMax(tf.add(tf.dot(inputTensor, models[selected_index]['coef_']), models[selected_index]['intercept_']), 1);
        console.log(result_action.dataSync());
        return result_action.dataSync()[0];
    }
}

export default LimePrediction;