/**
 * compute the energy of the current state
 * @param {*} network
 */
function energy(network) {
    const W = network.W,
          V = network.V,
          θ = network.θ;
    return W.mul(outer(V, V)).div(-2).sum() - V.mul(θ).sum();
}

/**
 * settle the network in a low energy state
 * @param {*} network
 * @param {*} iterations
 */
function settle(network, iterations) {
    for(let i=0; i < iterations; i++) {
        //network = update(network);
        network = updateStochasic(network);
    }
    return network;
}

/**
 * Step toward a fixed point in the Hopfield Network
 * by reducing the energy in the Hopfield Network
 * @param {HopfieldNetwork} network
 */
function update(network) {
    // V <=  1 if WV > θ else -1
    const W = network.W,
          V = network.V,
          θ = network.θ;
    network.V = where(greaterEqual(dot(W, V),θ), 1, -1);
    return network;
}

/**
 * Step toward a fixed point in the Hopfield Network stochastically
 * by reducing the energy in the Hopfield Network
 * @param {HopfieldNetwork} network
 */
function updateStochasic(network) {
    // V <=  1 if WV > θ else -1
    const W = network.W,
          V = network.V,
          θ = network.θ;
    const neuronIdx = Math.floor(Math.random()*V.size);
    network.V.set([neuronIdx], where(greaterEqual(dot(W, V), θ), 1, -1).get([neuronIdx]));
    return network;
}

/**
 * Step toward a fixed point in the Hopfield Network stochastically
 * by reducing the energy in the Hopfield Network
 * @param {HopfieldNetwork} network
 */
function updateStochasicI(network, i) {
    // V <=  1 if WV > θ else -1
    const W = network.W,
          V = network.V,
          θ = network.θ;
    network.V.set([i], where(greaterEqual(dot(W, V), θ), 1, -1).get([i]));
    return network;
}

/**
 * Learn to store the patterns as fixed points using a Hebbian Learning scheme
 * @param {Tensor} patterns
 * @param {HopfieldNetwork} network
 * @returns {Tensor} W a weight matrix
 */
function learnHebbian(patterns, network) {
    let M = patterns.length;
    let N = network.V.size;
    let W = fillT([N, N], 0);
    console.log(`${M} ${N}`);
    for (let i=0; i < M; i++) {
        let pattern = patterns[i];
        W = W.add(outer(pattern, pattern));
    }
    network.W = W.div(M);
    //set diagonal to zero
    let s = network.W.shape;
    network.W.A = network.W.A.map((w, i) => i % s[0] === Math.floor(i / s[0]) ? 0 : w);
    return network;
}

////////////////////////////////////////////////////////////////////////////////
// fastish math stuff
////////////////////////////////////////////////////////////////////////////////
class Tensor {
    constructor(T) {
        this.shape = shape(T);
        this.A     = flatten(T); // a flat, and optimized array
        this.size  = product(this.shape);

        this.initIndexStuff();
    }
    get(index) {
        return this.A[this.getIndex(index)];
    }
    set(index, value) {
        this.A[this.getIndex(index)] = value;
    }
    getIndex(index) {
        if (greater(index, this.shape).some(x => x) || less(index,0).some(x => x)) {
            throw `index ${index} not compatible with ${this.shape}`;
        }
        return sum(mul(this.indexHelper, index));
    }
    initIndexStuff() {
        this.indexHelper = fillArray(this.shape.length, 0);
        let accum = 1;
        for(let i=this.indexHelper.length - 1; i >=0; i--) {
            this.indexHelper[i] = accum;
            accum = accum*this.shape[i];
        }
    }

    reshape(shape) {
        this.shape = shape;
        this.initIndexStuff();
        return this;
    }

    map(f) {
        this.A = this.A.map(f);
        return this;
    }

    less(B) {
        this.A = less(this.A, B);
        return this;
    }

    T() {
        this.shape = this.shape.reverse();
        return this;
    }

    div(B) {
        if (B.constructor.name === "Tensor") {
            return new Tensor(div(this.A, B.A)).reshape(this.shape);
        }
        return new Tensor(div(this.A, B)).reshape(this.shape);
    }

    mul(B) {
        if (B.constructor.name === "Tensor") {
            return new Tensor(mul(this.A, B.A)).reshape(this.shape);
        }
        return new Tensor(mul(this.A, B)).reshape(this.shape);
    }

    add(B) {
        if (B.constructor.name === "Tensor") {
            return new Tensor(add(this.A, B.A)).reshape(this.shape);
        }
        return new Tensor(add(this.A, B)).reshape(this.shape);
    }

    sub(B) {
        if (B.constructor.name === "Tensor") {
            return new Tensor(sub(this.A, B.A)).reshape(this.shape);
        }
        return new Tensor(sub(this.A, B)).reshape(this.shape);
    }


    greaterEqual(B) {
        return new Tensor(greaterEqual(this.A, B));
    }

   sum() {
       return sum(this.A);
   }

}

function fillArray(n, x) {
    let array = new Array(n);
    if (typeof n === 'number') {
        if (typeof x === 'function') {
            for (let i=0; i<n; i++) {
                array[i] = x(i);
            }
        } else {
            array = array.fill(x);
        }
    }
    return array;
}

function fillT(N, X) {
    function fillHelper(n, x) {
        let tensor = (Array.isArray(n)) ? new Array(n[0]) : new Array(n);
        if (typeof n === 'number') {
            if (typeof x === 'function') {
                for (let i=0; i<n; i++) {
                    tensor[i] = x(i);
                }
            } else {
                tensor = tensor.fill(x);
            }
        } else if (Array.isArray(n)) {
            if (n.length == 1) {
                if (typeof x === 'function') {
                    for (let i=0; i<n; i++) {
                        tensor[i] = x(i);
                    }
                } else {
                    tensor = tensor.fill(x);
                }
            } else if (n.length > 1) {
                tensor.fill(x);
                for (let i=0; i < n[0]; i++) {
                    tensor[i] = fillHelper(tail(n), x);
                }
            }
        }
        return tensor;
    }
    return new Tensor(fillHelper(N, X));
}

function shape(A) {
    function shapeHelper(A, s) {
        if (Array.isArray(A)) {
            return shapeHelper(A[0], s.concat(A.length));
        }
        return s;
    }
    return shapeHelper(A, []);
}

// This is done in a linear time O(n) without recursion
// memory complexity is O(1) or O(n) if mutable param is set to false
function flatten(array, mutable) {
    var toString = Object.prototype.toString;
    var arrayTypeStr = '[object Array]';

    var result = [];
    var nodes = (mutable && array) || array.slice();
    var node;

    if (!array.length) {
        return result;
    }

    node = nodes.pop();

    do {
        if (toString.call(node) === arrayTypeStr) {
            nodes.push.apply(nodes, node);
        } else {
            result.push(node);
        }
    } while (nodes.length && (node = nodes.pop()) !== undefined);

    result.reverse(); // we reverse result to restore the original order
    return result;
}

function outer(A, B) {
    let C = fillT([A.size, B.size], 0);
    for (let i=0; i < A.size; i++) {
        for (let j=0; j < B.size; j++) {
            C.set([i, j], A.A[i]*B.A[j]);
        }
    }
    return C;
}

function dot(A, B) {
    const shape = init(A.shape).concat(tail(B.shape));
    //console.log("(" + A.shape + ") * (" + B.shape + ") => (" + shape + ")");
    if (shape.length === 0) {
        return sum(mul(A.A, B.A).mul(B.A));
    }
    //
    let M = last(A.shape);
    let C = fillT(shape, 0);
    let index=fillArray(C.shape.length, 0);
    for (let i =0; i<C.size; i++) {
        let value = 0;
        let idxA = index.slice(0, A.shape.length-1).concat([0]);
        let idxB = [0].concat(index.slice(A.shape.length - 1, index.length));
        for (let j=0; j<M; j++) {
            idxA[idxA.length -1] = j;
            idxB[0] = j;
            value += A.get(idxA)*B.get(idxB);
        }
        C.set(index, value);
        index = nextIndex(index, C.shape);
    }
    return C
}

function nextIndex(index, shape) {
    index[index.length - 1] += 1;
    let flag = false;
    let i = index.length - 1;
    while (index[i] >= shape[i]) {
        flag = true;
        index[i] = 0;
        i--;
        index[i]++;
    }
    if (index.every(x => x === 0)) {
        return false;
    }
    return index;
}
function add(a, b) {
    if (typeof b === 'number') {
        return this.map((a) => a + b);
    }
    return a.map((a, i) => a + b[i]);
};

function last(a) {
    if (a.length > 0) {
        return a[a.length - 1];
    }
};

function tail(a) {
    if (a.length > 0) {
        return a.slice(1, a.length);
    }
};

function init(a) {
    if (a.length > 0) {
        return a.slice(0, a.length - 1);
    }
};

function sub(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a - b);
    }
    return a.map((a, i) => a - b[i]);
};

function delta(a, b) {
    if (typeof b === 'number') {
        for(let i=0; i < a.length; i++) {
            a[i] += b;
        }
    }
    for(let i=0; i < a.length; i++) {
        a[i] += b[i];
    }

}

function mul(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a*b);
    }
    return a.map((a, i) => a*b[i]);
};

function div(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a/b);
    }
    return a.map((a, i) => a/b[i]);
};

function or(a, b) {
    return a.map((a, i) => a || b[i]);
};

function neg(a) {
    return a.map((a) => -a);
};

function exp(a) {
    return a.map((a) => Math.exp(a));
};

function sum(a) {
    return a.reduce((prev, curr) => prev + curr);
}

function product(a) {
    return a.reduce((prev, curr) => prev * curr);
}

function max(a, B) {
    return a.map((a, i) => Math.max(B[i], a));
}

function less(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a < b);
    }
    return a.map((a, i) => a < b[i])
}

function greaterEqual(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a >= b);
    }
    return a.map((a, i) => a >= b[i])
}

function greater(a, b) {
    if (typeof b === 'number') {
        return a.map((a) => a > b);
    }
    return a.map((a, i) => a > b[i])
}

function equivalent(a, b) {
    return a.every((a, i) => a === b[i])
}

function where(M, a, b) {
    let getA = (typeof a === "number") ? (i) => a : (i) => a[i];
    let getB = (typeof b === "number") ? (i) => b : (i) => b[i];
    if (M.constructor.name === "Tensor") {
        M.A = where(M.A, a, b);
        return M;
    }
    return M.map((m, i) => (m) ? getA(i) : getB(i));
};

function randomUniform(N) {
    return fillT(N, () => Math.random());
}
