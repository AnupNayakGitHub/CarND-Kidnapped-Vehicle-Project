/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    //Set the number of particles to 1024
    num_particles = 1024;


    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
        //TODO: not initialized weights vector yet.

        cout << "Measurements : "<< p.id << ", " << p.x << ", " << p.y << ", " << p.theta << endl;
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);
    double d_theta = yaw_rate * delta_t;
    for (std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
        //cout << "ID is : " << it->id <<endl;
        if(d_theta == 0) {
            it->x += velocity * cos(it->theta) + dist_x(gen);
            it->y += velocity * sin(it->theta) + dist_y(gen);
            it->theta += dist_theta(gen);
        }
        else {
            it->x += (velocity * (sin(it->theta + d_theta) - sin(it->theta))/yaw_rate) + dist_x(gen);
            it->y += (velocity * (cos(it->theta) - cos(it->theta + d_theta))/yaw_rate) + dist_y(gen);
            it->theta += d_theta + dist_theta(gen);
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    //cout << "dataAssociation started ...." << endl;
    //cout << "Predicted size : " << predicted.size() << endl;
    for(auto& observed_lm : observations) {
        int min_id = 0;
        double min_dist_sq = std::numeric_limits<double>::max();
        for(const auto& real_lm : predicted){
            //
            double dist_sq  = ((real_lm.x-observed_lm.x)*(real_lm.x-observed_lm.x) +
                                (real_lm.y-observed_lm.y)*(real_lm.y-observed_lm.y));
            //cout << "Squared distance : " << dist_sq << " prediction id : " << real_lm.id << endl;
            if (dist_sq < min_dist_sq) {
                min_id = real_lm.id;
                min_dist_sq = dist_sq;
            }
        }
        observed_lm.id = min_id;
        //assert(observed_lm.id != 0);
    }
    //for(const auto& observation: observations){
    //    cout << "Observation : " << observation.id << ", " << observation.x << ", " << observation.y << endl;
    //}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (auto& particle : particles) {
        //cout << "Particle " << particle.id << " " << particle.x << " " << particle.y << " " << particle.theta << endl;
        //Rotate and translate observations
        vector<LandmarkObs> global_observations;
        for (auto observation: observations) {
            LandmarkObs lm;
            lm.id = 0;
            lm.x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x; 
            lm.y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y; 
            global_observations.push_back(lm);
        }

        //Find qualified landmarks
        vector<LandmarkObs> global_qualified_lms;
        double allowed_sq_dist(sensor_range*sensor_range);
        for(auto lm : map_landmarks.landmark_list){
            if (pow(particle.x - lm.x_f,2) + pow(particle.y - lm.y_f,2) <= allowed_sq_dist) {
                LandmarkObs observable_lm;
                observable_lm.id = lm.id_i;
                observable_lm.x = lm.x_f;
                observable_lm.y = lm.y_f;
                global_qualified_lms.push_back(observable_lm);
            }
        }

        //Associate data
        dataAssociation(global_qualified_lms, global_observations);

        //get the probability
        double x_var = std_landmark[0]*std_landmark[0];
        double y_var = std_landmark[1]*std_landmark[1];
        double normalizer = 2.0*M_PI*std_landmark[0]*std_landmark[1];

        particle.weight = 1;
        for(auto& global_observation : global_observations){
            if (global_observation.id == 0){
                particle.weight = 0;
                break;
            }

            Map::single_landmark_s fixed_lm = map_landmarks.landmark_list[global_observation.id - 1];
            //asserted to make sure the assumption for indexing is correct!
            //if(fixed_lm.id_i != global_observation.id) {
            //    cout << "Fixed landmark id " << fixed_lm.id_i << " Global Observation Id " << global_observation.id << endl;
            //}
            assert(fixed_lm.id_i == global_observation.id);
            double dx_sq = pow(fixed_lm.x_f - global_observation.x,2);
            double dy_sq = pow(fixed_lm.y_f - global_observation.y,2);
            double exponent = ((dx_sq/x_var)+(dy_sq/y_var))/(-2);
            particle.weight *= exp(exponent)/normalizer;
            //cout << "Particle " << particle.id << " w/ updated weight " << particle.weight << endl;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    //Find the best particle
    Particle best_particle;
    best_particle.weight = -1;
    for(const auto& particle : particles) {
        if(particle.weight > best_particle.weight)
            best_particle = particle;
    }
    if (best_particle.weight == 0) {
        cout << "Best particle " << best_particle.weight << " " << best_particle.x << " " << best_particle.y << " " << best_particle.theta << endl;
        for(auto& particle : particles) {
            particle.weight = 1;
        }
        return;
    }

    double weight_offset(best_particle.weight * 2);
    //int index = rand()%num_particles;
    int index = 0;
    //cout <<"The starting index is " << index << endl;

    vector<Particle> new_particles;
    while(new_particles.size() < num_particles){
        Particle p = particles[index];
        if (p.weight >= weight_offset) {
            //copy this particle
            weight_offset += best_particle.weight * 2 - p.weight;
            p.id = new_particles.size();
            new_particles.push_back(p);
        }
        else {
            weight_offset -= p.weight;
        }
        index = (++index%num_particles);
    }
    //for (auto& particle : new_particles) {
    //    cout << "New Particle " << particle.id << " " << particle.x << " " << particle.y << " " << particle.theta << " " << particle.weight << endl;
    //}
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
